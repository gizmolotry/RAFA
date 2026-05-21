from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RUNTIME = Path(__file__).resolve().parent
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from resonant_law_objects import (  # noqa: E402
    LAW_OBJECT_FEATURE_NAMES,
    law_object_feature_vector,
    load_law_objects_from_json,
    object_family_value,
    synthetic_law_objects,
)


class ContrastiveLawEmbedding(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int = 32, hidden_dim: int = 96) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.safety_head = nn.Sequential(
            nn.Linear(embedding_dim, max(16, embedding_dim)),
            nn.GELU(),
            nn.Linear(max(16, embedding_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = F.normalize(self.encoder(features), dim=-1)
        safety = torch.sigmoid(self.safety_head(embedding)).squeeze(-1)
        return embedding, safety


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_list(values: Any) -> list[float]:
    if isinstance(values, (list, tuple)):
        out: list[float] = []
        for item in values:
            if isinstance(item, (list, tuple)):
                out.extend(_safe_list(item))
            else:
                out.append(_as_float(item))
        return out
    if values is None:
        return []
    return [_as_float(values)]


def _normalize(values: list[float]) -> list[float]:
    raw = [abs(float(x)) for x in values]
    if not raw:
        return [1.0]
    total = sum(raw)
    if total <= 1.0e-12:
        return [1.0 / len(raw)] * len(raw)
    return [x / total for x in raw]


def _cosine01(left: list[float], right: list[float]) -> float:
    n = min(len(left), len(right))
    if n <= 0:
        return 0.0
    a = left[:n]
    b = right[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return 0.0
    return _clamp01(0.5 + 0.5 * dot / (na * nb))


def _window_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    l0 = _as_float(left.get("window_start", 0.0))
    l1 = _as_float(left.get("window_end", 1.0))
    r0 = _as_float(right.get("window_start", 0.0))
    r1 = _as_float(right.get("window_end", 1.0))
    inter = max(0.0, min(l1, r1) - max(l0, r0))
    union = max(1.0e-8, max(l1, r1) - min(l0, r0))
    return _clamp01(inter / union)


def recurrence_compatibility(left: dict[str, Any], right: dict[str, Any]) -> float:
    """No structural-family teacher: score cross-case recurrence from continuous law geometry only."""
    lk = dict(left.get("key", left.get("query", {})) or {})
    rk = dict(right.get("key", right.get("query", {})) or {})
    q_score = _cosine01(_normalize(_safe_list(lk.get("q_profile"))), _normalize(_safe_list(rk.get("q_profile"))))
    arc_score = 1.0 - abs(_as_float(lk.get("major_arc_proxy", 0.0)) - _as_float(rk.get("major_arc_proxy", 0.0)))
    residue_score = 1.0 - abs(_as_float(lk.get("minor_residue_proxy", 0.0)) - _as_float(rk.get("minor_residue_proxy", 0.0)))
    support_score = 1.0 - abs(_as_float(lk.get("support_mean", 0.0)) - _as_float(rk.get("support_mean", 0.0)))
    support_overlap = _window_overlap(lk, rk)
    coherence_score = 1.0 - abs(_as_float(lk.get("coherence_mean", 0.0)) - _as_float(rk.get("coherence_mean", 0.0)))
    boundary_score = 1.0 - abs(_as_float(lk.get("boundary_score", 0.5)) - _as_float(rk.get("boundary_score", 0.5)))
    temporal_score = 1.0 - 0.65 * abs(_as_float(lk.get("temporal_center", 0.5)) - _as_float(rk.get("temporal_center", 0.5)))
    temporal_score -= 0.35 * abs(_as_float(lk.get("temporal_width", 1.0)) - _as_float(rk.get("temporal_width", 1.0)))
    generation_score = 1.0 - min(1.0, abs(_as_float(left.get("generation", 1.0)) - _as_float(right.get("generation", 1.0))) / 4.0)
    kind_score = 1.0 if str(left.get("object_kind", "")) == str(right.get("object_kind", "")) else 0.0
    score = (
        0.24 * q_score
        + 0.12 * _clamp01(arc_score)
        + 0.12 * _clamp01(residue_score)
        + 0.10 * _clamp01(support_score)
        + 0.10 * support_overlap
        + 0.08 * _clamp01(coherence_score)
        + 0.08 * _clamp01(boundary_score)
        + 0.08 * _clamp01(temporal_score)
        + 0.04 * _clamp01(generation_score)
        + 0.04 * kind_score
    )
    return _clamp01(score)


def operator_safety_proxy(obj: dict[str, Any]) -> float:
    key = dict(obj.get("key", obj.get("query", {})) or {})
    value = dict(obj.get("value", {}) or {})
    support = _as_float(value.get("support_mass", key.get("support_mean", 0.0)))
    coherence = _as_float(value.get("coherence_mass", key.get("coherence_mean", 0.0)))
    boundary = _as_float(key.get("boundary_score", 0.5))
    residue_safety = 1.0 - _as_float(key.get("minor_residue_proxy", 1.0))
    budget = min(1.0, _as_float(value.get("writeback_budget", 0.0)) / 2.0)
    ready = _as_float(value.get("causal_use_ready", 0.0))
    return _clamp01(0.24 * support + 0.24 * coherence + 0.20 * boundary + 0.16 * residue_safety + 0.08 * budget + 0.08 * ready)


def _expand_inputs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(path.rglob("nested_commitment_summary.json")))
            out.extend(sorted(path.rglob("nested_commitment_report.json")))
            continue
        if path.exists():
            out.append(path)
    dedup: dict[str, Path] = {}
    for path in out:
        dedup[str(path.resolve()).lower()] = path
    return list(dedup.values())


def _load_objects(inputs: list[str], *, synthetic_smoke: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    objects: list[dict[str, Any]] = []
    source_files: list[str] = []
    for path in _expand_inputs(inputs):
        loaded = load_law_objects_from_json(path)
        if loaded:
            source_files.append(str(path))
            objects.extend(loaded)
    if synthetic_smoke or not objects:
        source_files.append("synthetic_smoke")
        base = synthetic_law_objects()
        for case in ("synthetic_a", "synthetic_b", "synthetic_c"):
            for obj in base:
                clone = json.loads(json.dumps(obj))
                old_id = str(clone.get("object_id", "synthetic"))
                old_family = str(clone.get("family_key", "synthetic:family"))
                clone["case"] = case
                clone["object_id"] = f"{case}:{old_id}"
                clone["local_family_key"] = f"{case}:{old_family}"
                clone["family_key"] = clone["local_family_key"]
                objects.append(clone)
    dedup: dict[str, dict[str, Any]] = {}
    for obj in objects:
        object_id = str(obj.get("object_id", ""))
        if object_id:
            dedup.setdefault(object_id, obj)
    return list(dedup.values()), source_files


def _case_split(objects: list[dict[str, Any]], holdout_case: str | None) -> tuple[list[int], list[int], str]:
    cases = sorted({str(obj.get("case", "unknown_case")) for obj in objects})
    holdout = str(holdout_case or (cases[-1] if cases else "unknown_case"))
    if holdout not in cases and cases:
        holdout = cases[-1]
    train = [idx for idx, obj in enumerate(objects) if str(obj.get("case", "unknown_case")) != holdout]
    test = [idx for idx, obj in enumerate(objects) if str(obj.get("case", "unknown_case")) == holdout]
    if not train:
        split = max(1, len(objects) // 2)
        train = list(range(split))
        test = list(range(split, len(objects))) or list(range(split))
    return train, test, holdout


def _sample_recurrence_pairs(
    objects: list[dict[str, Any]],
    train_indices: list[int],
    *,
    positive_threshold: float,
    negative_threshold: float,
    positives_per_query: int,
    hard_negatives_per_query: int,
    pair_limit: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[float], dict[str, float]]:
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    labels: list[float] = []
    pos_count = 0
    neg_count = 0
    high_scoring_negatives = 0
    indices = list(train_indices)
    rng.shuffle(indices)
    for left in indices:
        left_case = str(objects[left].get("case", ""))
        candidates = [
            idx
            for idx in train_indices
            if idx != left
            and str(objects[idx].get("case", "")) != left_case
            and str(objects[idx].get("object_kind", "")) == str(objects[left].get("object_kind", ""))
        ]
        if not candidates:
            continue
        scored = [(idx, recurrence_compatibility(objects[left], objects[idx])) for idx in candidates]
        scored.sort(key=lambda row: row[1], reverse=True)
        positives = [idx for idx, score in scored if score >= positive_threshold][: max(1, int(positives_per_query))]
        negatives = [idx for idx, score in scored if score <= negative_threshold]
        hard_negative_pool = [idx for idx, score in scored if negative_threshold < score < positive_threshold]
        rng.shuffle(negatives)
        for right in positives:
            pairs.append((left, right))
            labels.append(1.0)
            pos_count += 1
            if len(pairs) >= pair_limit:
                break
        if len(pairs) >= pair_limit:
            break
        chosen_hard = hard_negative_pool[: max(0, int(hard_negatives_per_query))]
        chosen_easy = negatives[: max(0, int(hard_negatives_per_query) - len(chosen_hard))]
        for right in chosen_hard + chosen_easy:
            pairs.append((left, right))
            labels.append(0.0)
            neg_count += 1
            if right in chosen_hard:
                high_scoring_negatives += 1
            if len(pairs) >= pair_limit:
                break
        if len(pairs) >= pair_limit:
            break
    order = list(range(len(pairs)))
    rng.shuffle(order)
    stats = {
        "positive_pair_count": float(pos_count),
        "negative_pair_count": float(neg_count),
        "high_scoring_negative_count": float(high_scoring_negatives),
    }
    return [pairs[idx] for idx in order], [labels[idx] for idx in order], stats


def _feature_tensor(objects: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([law_object_feature_vector(obj) for obj in objects], dtype=torch.float32, device=device)


def _safety_tensor(objects: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([operator_safety_proxy(obj) for obj in objects], dtype=torch.float32, device=device)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _train_embedding(
    model: ContrastiveLawEmbedding,
    features: torch.Tensor,
    safety_targets: torch.Tensor,
    pairs: list[tuple[int, int]],
    labels: list[float],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    safety_weight: float,
) -> dict[str, float]:
    if not pairs:
        return {"final_loss": 0.0, "final_contrastive_loss": 0.0, "final_safety_loss": 0.0}
    device = features.device
    pair_tensor = torch.tensor(pairs, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-3)
    last = {"final_loss": 0.0, "final_contrastive_loss": 0.0, "final_safety_loss": 0.0}
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(pair_tensor.size(0), device=device)
        losses: list[float] = []
        c_losses: list[float] = []
        s_losses: list[float] = []
        for start in range(0, int(pair_tensor.size(0)), max(1, int(batch_size))):
            batch = perm[start : start + max(1, int(batch_size))]
            idx = pair_tensor[batch]
            embeddings, safety_pred = model(features)
            sim = (embeddings[idx[:, 0]] * embeddings[idx[:, 1]]).sum(dim=-1) / max(1.0e-6, float(temperature))
            contrastive_loss = F.binary_cross_entropy_with_logits(sim, label_tensor[batch])
            endpoint_idx = torch.unique(idx.flatten())
            safety_loss = F.mse_loss(safety_pred[endpoint_idx], safety_targets[endpoint_idx])
            loss = contrastive_loss + float(safety_weight) * safety_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            c_losses.append(float(contrastive_loss.detach().cpu().item()))
            s_losses.append(float(safety_loss.detach().cpu().item()))
        last = {
            "final_loss": _mean(losses),
            "final_contrastive_loss": _mean(c_losses),
            "final_safety_loss": _mean(s_losses),
        }
    return last


def _evaluate_embedding(
    model: ContrastiveLawEmbedding,
    objects: list[dict[str, Any]],
    features: torch.Tensor,
    safety_targets: torch.Tensor,
    query_indices: list[int],
    candidate_indices: list[int],
    *,
    batch_size: int,
    positive_threshold: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        embeddings, safety_pred = model(features)
        for q_idx in query_indices:
            q_case = str(objects[q_idx].get("case", ""))
            candidates = [
                idx
                for idx in candidate_indices
                if idx != q_idx
                and str(objects[idx].get("case", "")) != q_case
                and str(objects[idx].get("object_kind", "")) == str(objects[q_idx].get("object_kind", ""))
            ]
            positives = [
                idx
                for idx in candidates
                if recurrence_compatibility(objects[q_idx], objects[idx]) >= positive_threshold
            ]
            if not candidates or not positives:
                continue
            q = embeddings[q_idx].view(1, -1)
            scores: list[tuple[int, float]] = []
            for start in range(0, len(candidates), max(1, int(batch_size))):
                batch = candidates[start : start + max(1, int(batch_size))]
                cand = embeddings[torch.tensor(batch, dtype=torch.long, device=features.device)]
                sim = (q.expand(cand.size(0), -1) * cand).sum(dim=-1).detach().cpu().tolist()
                scores.extend((idx, float(score)) for idx, score in zip(batch, sim))
            scores.sort(key=lambda row: row[1], reverse=True)
            top_idx, top_score = scores[0]
            best_pos = max((score for idx, score in scores if idx in positives), default=-1.0)
            best_decoy = max((score for idx, score in scores if idx not in positives), default=-1.0)
            rows.append(
                {
                    "query_object_id": str(objects[q_idx].get("object_id", "")),
                    "top_object_id": str(objects[top_idx].get("object_id", "")),
                    "query_case": q_case,
                    "top_case": str(objects[top_idx].get("case", "")),
                    "recurrence_positive_hit": 1.0 if top_idx in positives else 0.0,
                    "same_structural_family": 1.0
                    if object_family_value(objects[q_idx], "structural_family_key")
                    == object_family_value(objects[top_idx], "structural_family_key")
                    else 0.0,
                    "same_local_family": 1.0
                    if object_family_value(objects[q_idx], "local_family_key")
                    == object_family_value(objects[top_idx], "local_family_key")
                    else 0.0,
                    "top_score": top_score,
                    "best_positive_score": best_pos,
                    "best_decoy_score": best_decoy,
                    "embedding_margin": best_pos - best_decoy,
                    "top_recurrence_compatibility": recurrence_compatibility(objects[q_idx], objects[top_idx]),
                    "query_safety_target": float(safety_targets[q_idx].detach().cpu().item()),
                    "top_safety_target": float(safety_targets[top_idx].detach().cpu().item()),
                    "query_safety_pred": float(safety_pred[q_idx].detach().cpu().item()),
                    "top_safety_pred": float(safety_pred[top_idx].detach().cpu().item()),
                }
            )
    return {
        "query_count": len(rows),
        "top1_recurrence_accuracy": _mean([float(row["recurrence_positive_hit"]) for row in rows]),
        "top1_structural_family_accuracy": _mean([float(row["same_structural_family"]) for row in rows]),
        "top1_local_leakage": _mean([float(row["same_local_family"]) for row in rows]),
        "mean_embedding_margin": _mean([float(row["embedding_margin"]) for row in rows]),
        "mean_top_recurrence_compatibility": _mean([float(row["top_recurrence_compatibility"]) for row in rows]),
        "mean_query_safety_target": _mean([float(row["query_safety_target"]) for row in rows]),
        "mean_top_safety_target": _mean([float(row["top_safety_target"]) for row in rows]),
        "mean_abs_safety_error": _mean(
            [abs(float(row["query_safety_target"]) - float(row["query_safety_pred"])) for row in rows]
        ),
        "rows": rows,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    holdout = payload["holdout_retrieval"]
    all_eval = payload["all_case_retrieval"]
    lines = [
        "# Contrastive Structural Embedding Probe",
        "",
        "This assay trains a law-object embedding from cross-case recurrence compatibility and operator-safety proxy targets. It does not use `structural_family_key` as a training label; that key is only used afterward as an evaluation yardstick.",
        "",
        "## Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- object count: `{summary['object_count']}`",
        f"- train objects: `{summary['train_object_count']}`",
        f"- holdout objects: `{summary['holdout_object_count']}`",
        f"- holdout case: `{summary['holdout_case']}`",
        f"- pair count: `{summary['pair_count']}`",
        f"- positive pairs: `{summary['positive_pair_count']}`",
        f"- negative pairs: `{summary['negative_pair_count']}`",
        f"- final contrastive loss: `{summary['final_contrastive_loss']}`",
        f"- final safety loss: `{summary['final_safety_loss']}`",
        f"- checkpoint: `{payload.get('checkpoint_path', '')}`",
        "",
        "## Holdout Retrieval",
        "",
        f"- query count: `{holdout['query_count']}`",
        f"- top1 recurrence accuracy: `{holdout['top1_recurrence_accuracy']}`",
        f"- top1 structural-family accuracy: `{holdout['top1_structural_family_accuracy']}`",
        f"- top1 local leakage: `{holdout['top1_local_leakage']}`",
        f"- mean embedding margin: `{holdout['mean_embedding_margin']}`",
        f"- mean top recurrence compatibility: `{holdout['mean_top_recurrence_compatibility']}`",
        f"- mean abs safety error: `{holdout['mean_abs_safety_error']}`",
        "",
        "## All-Case Retrieval",
        "",
        f"- query count: `{all_eval['query_count']}`",
        f"- top1 recurrence accuracy: `{all_eval['top1_recurrence_accuracy']}`",
        f"- top1 structural-family accuracy: `{all_eval['top1_structural_family_accuracy']}`",
        f"- top1 local leakage: `{all_eval['top1_local_leakage']}`",
        f"- mean embedding margin: `{all_eval['mean_embedding_margin']}`",
        "",
        "## Sources",
        "",
    ]
    for source in payload.get("source_files", []):
        lines.append(f"- `{source}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
    inputs: list[str],
    out_dir: Path,
    synthetic_smoke: bool,
    holdout_case: str | None,
    positive_threshold: float,
    negative_threshold: float,
    positives_per_query: int,
    hard_negatives_per_query: int,
    pair_limit: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    embedding_dim: int,
    temperature: float,
    safety_weight: float,
    seed: int,
    device_name: str,
) -> dict[str, str]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    objects, source_files = _load_objects(inputs, synthetic_smoke=synthetic_smoke)
    features = _feature_tensor(objects, device)
    safety_targets = _safety_tensor(objects, device)
    train_idx, test_idx, holdout = _case_split(objects, holdout_case)
    pairs, labels, pair_stats = _sample_recurrence_pairs(
        objects,
        train_idx,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        positives_per_query=positives_per_query,
        hard_negatives_per_query=hard_negatives_per_query,
        pair_limit=pair_limit,
        seed=seed,
    )
    model = ContrastiveLawEmbedding(features.size(1), embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    train_summary = _train_embedding(
        model,
        features,
        safety_targets,
        pairs,
        labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=temperature,
        safety_weight=safety_weight,
    )
    holdout_retrieval = _evaluate_embedding(
        model,
        objects,
        features,
        safety_targets,
        test_idx,
        train_idx,
        batch_size=batch_size,
        positive_threshold=positive_threshold,
    )
    all_indices = list(range(len(objects)))
    all_case_retrieval = _evaluate_embedding(
        model,
        objects,
        features,
        safety_targets,
        all_indices,
        all_indices,
        batch_size=batch_size,
        positive_threshold=positive_threshold,
    )
    summary = {
        "status": "pass_contrastive_recurrence_signal"
        if holdout_retrieval["top1_recurrence_accuracy"] > 0.0
        else "fail_no_contrastive_recurrence_signal",
        "object_count": len(objects),
        "train_object_count": len(train_idx),
        "holdout_object_count": len(test_idx),
        "holdout_case": holdout,
        "pair_count": len(pairs),
        **pair_stats,
        **train_summary,
    }
    payload = {
        "summary": summary,
        "source_files": source_files,
        "feature_names": list(LAW_OBJECT_FEATURE_NAMES),
        "holdout_retrieval": {k: v for k, v in holdout_retrieval.items() if k != "rows"},
        "holdout_rows": holdout_retrieval["rows"],
        "all_case_retrieval": {k: v for k, v in all_case_retrieval.items() if k != "rows"},
        "all_case_rows": all_case_retrieval["rows"],
        "config": {
            "holdout_case": holdout,
            "positive_threshold": float(positive_threshold),
            "negative_threshold": float(negative_threshold),
            "positives_per_query": int(positives_per_query),
            "hard_negatives_per_query": int(hard_negatives_per_query),
            "pair_limit": int(pair_limit),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "hidden_dim": int(hidden_dim),
            "embedding_dim": int(embedding_dim),
            "temperature": float(temperature),
            "safety_weight": float(safety_weight),
            "seed": int(seed),
            "device": str(device),
            "teacher": "cross_case_recurrence_compatibility_not_structural_family_key",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "contrastive_structural_embedding_probe.json"
    md_path = out_dir / "CONTRASTIVE_STRUCTURAL_EMBEDDING_PROBE.md"
    checkpoint_path = out_dir / "contrastive_structural_embedding_probe.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": list(LAW_OBJECT_FEATURE_NAMES),
            "config": payload["config"],
            "summary": summary,
        },
        checkpoint_path,
    )
    payload["checkpoint_path"] = str(checkpoint_path)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    return {"json": str(json_path), "markdown": str(md_path), "status": str(summary["status"])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a contrastive law-object embedding from recurrence and safety targets.")
    ap.add_argument("--input", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synthetic-smoke", action="store_true")
    ap.add_argument("--holdout-case", default=None)
    ap.add_argument("--positive-threshold", type=float, default=0.94)
    ap.add_argument("--negative-threshold", type=float, default=0.86)
    ap.add_argument("--positives-per-query", type=int, default=3)
    ap.add_argument("--hard-negatives-per-query", type=int, default=3)
    ap.add_argument("--pair-limit", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--embedding-dim", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.12)
    ap.add_argument("--safety-weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(
        json.dumps(
            run_probe(
                inputs=[str(item) for item in args.input],
                out_dir=Path(args.out_dir),
                synthetic_smoke=bool(args.synthetic_smoke),
                holdout_case=args.holdout_case,
                positive_threshold=float(args.positive_threshold),
                negative_threshold=float(args.negative_threshold),
                positives_per_query=int(args.positives_per_query),
                hard_negatives_per_query=int(args.hard_negatives_per_query),
                pair_limit=int(args.pair_limit),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                hidden_dim=int(args.hidden_dim),
                embedding_dim=int(args.embedding_dim),
                temperature=float(args.temperature),
                safety_weight=float(args.safety_weight),
                seed=int(args.seed),
                device_name=str(args.device),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
