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
    score_query_to_law_object,
    synthetic_law_objects,
)


class PairStructuralProbe(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.structural_head = nn.Linear(hidden_dim, 1)
        self.local_head = nn.Linear(hidden_dim, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair = torch.cat([torch.abs(left - right), left * right], dim=-1)
        hidden = self.net(pair)
        return self.structural_head(hidden).squeeze(-1), self.local_head(hidden).squeeze(-1)


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
        for case_idx, case in enumerate(("synthetic_a", "synthetic_b", "synthetic_c")):
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


def _feature_tensor(objects: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    rows = [law_object_feature_vector(obj) for obj in objects]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _case_split(objects: list[dict[str, Any]], holdout_case: str | None) -> tuple[list[int], list[int], str]:
    cases = sorted({str(obj.get("case", "unknown_case")) for obj in objects})
    if not cases:
        return [], [], "unknown_case"
    holdout = str(holdout_case or cases[-1])
    if holdout not in cases:
        holdout = cases[-1]
    train = [idx for idx, obj in enumerate(objects) if str(obj.get("case", "unknown_case")) != holdout]
    test = [idx for idx, obj in enumerate(objects) if str(obj.get("case", "unknown_case")) == holdout]
    if not train:
        split = max(1, len(objects) // 2)
        train = list(range(split))
        test = list(range(split, len(objects))) or list(range(split))
    return train, test, holdout


def _group_indices(objects: list[dict[str, Any]], indices: list[int], field: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx in indices:
        groups.setdefault(object_family_value(objects[idx], field), []).append(idx)
    return groups


def _sample_pairs(
    objects: list[dict[str, Any]],
    train_indices: list[int],
    *,
    pair_limit: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[float], list[float]]:
    rng = random.Random(seed)
    structural_groups = _group_indices(objects, train_indices, "structural_family_key")
    local_groups = _group_indices(objects, train_indices, "local_family_key")
    usable_structural = [members for members in structural_groups.values() if len(members) >= 2]
    pairs: list[tuple[int, int]] = []
    y_struct: list[float] = []
    y_local: list[float] = []
    half = max(1, pair_limit // 2)
    for _ in range(half):
        if not usable_structural:
            break
        members = rng.choice(usable_structural)
        left, right = rng.sample(members, 2)
        pairs.append((left, right))
        y_struct.append(1.0)
        y_local.append(1.0 if object_family_value(objects[left], "local_family_key") == object_family_value(objects[right], "local_family_key") else 0.0)
    all_indices = list(train_indices)
    attempts = 0
    while len(pairs) < pair_limit and attempts < pair_limit * 20 and len(all_indices) >= 2:
        attempts += 1
        left, right = rng.sample(all_indices, 2)
        same_struct = object_family_value(objects[left], "structural_family_key") == object_family_value(objects[right], "structural_family_key")
        if same_struct:
            continue
        pairs.append((left, right))
        y_struct.append(0.0)
        y_local.append(1.0 if object_family_value(objects[left], "local_family_key") == object_family_value(objects[right], "local_family_key") else 0.0)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    return [pairs[idx] for idx in order], [y_struct[idx] for idx in order], [y_local[idx] for idx in order]


def _train_probe(
    model: PairStructuralProbe,
    features: torch.Tensor,
    pairs: list[tuple[int, int]],
    y_struct: list[float],
    y_local: list[float],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, float]:
    if not pairs:
        return {"final_loss": 0.0, "final_structural_loss": 0.0, "final_local_loss": 0.0}
    device = features.device
    pair_tensor = torch.tensor(pairs, dtype=torch.long, device=device)
    y_struct_t = torch.tensor(y_struct, dtype=torch.float32, device=device)
    y_local_t = torch.tensor(y_local, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-3)
    last = {"final_loss": 0.0, "final_structural_loss": 0.0, "final_local_loss": 0.0}
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(pair_tensor.size(0), device=device)
        losses: list[float] = []
        s_losses: list[float] = []
        l_losses: list[float] = []
        for start in range(0, int(pair_tensor.size(0)), max(1, int(batch_size))):
            batch = perm[start : start + max(1, int(batch_size))]
            idx = pair_tensor[batch]
            s_logit, l_logit = model(features[idx[:, 0]], features[idx[:, 1]])
            s_loss = F.binary_cross_entropy_with_logits(s_logit, y_struct_t[batch])
            l_loss = F.binary_cross_entropy_with_logits(l_logit, y_local_t[batch])
            loss = s_loss + 0.35 * l_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            s_losses.append(float(s_loss.detach().cpu().item()))
            l_losses.append(float(l_loss.detach().cpu().item()))
        last = {
            "final_loss": _mean(losses),
            "final_structural_loss": _mean(s_losses),
            "final_local_loss": _mean(l_losses),
        }
    return last


def _geometric_score(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    return float(
        score_query_to_law_object(query, candidate, family_key_field="structural_family_key")["final_score"]
    )


def _evaluate_retrieval(
    model: PairStructuralProbe,
    objects: list[dict[str, Any]],
    features: torch.Tensor,
    query_indices: list[int],
    candidate_indices: list[int],
    *,
    batch_size: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    device = features.device
    model.eval()
    with torch.no_grad():
        for q_idx in query_indices:
            positives = [
                c_idx
                for c_idx in candidate_indices
                if c_idx != q_idx
                and object_family_value(objects[c_idx], "structural_family_key")
                == object_family_value(objects[q_idx], "structural_family_key")
            ]
            if not positives:
                continue
            scores: list[tuple[int, float, float]] = []
            q_feat = features[q_idx].view(1, -1)
            for start in range(0, len(candidate_indices), max(1, int(batch_size))):
                batch_indices = [idx for idx in candidate_indices[start : start + max(1, int(batch_size))] if idx != q_idx]
                if not batch_indices:
                    continue
                c_feat = features[torch.tensor(batch_indices, dtype=torch.long, device=device)]
                q_batch = q_feat.expand(c_feat.size(0), -1)
                s_logit, l_logit = model(q_batch, c_feat)
                s_prob = torch.sigmoid(s_logit).detach().cpu().tolist()
                l_prob = torch.sigmoid(l_logit).detach().cpu().tolist()
                scores.extend((idx, float(s), float(l)) for idx, s, l in zip(batch_indices, s_prob, l_prob))
            if not scores:
                continue
            scores.sort(key=lambda row: row[1], reverse=True)
            top_idx, top_score, top_local_prob = scores[0]
            best_positive = max((score for idx, score, _ in scores if idx in positives), default=0.0)
            best_decoy = max((score for idx, score, _ in scores if idx not in positives), default=0.0)
            query_case = str(objects[q_idx].get("case", ""))
            top_case = str(objects[top_idx].get("case", ""))
            rows.append(
                {
                    "query_object_id": str(objects[q_idx].get("object_id", "")),
                    "top_object_id": str(objects[top_idx].get("object_id", "")),
                    "query_case": query_case,
                    "top_case": top_case,
                    "same_case": 1.0 if query_case == top_case else 0.0,
                    "same_structural_family": 1.0
                    if object_family_value(objects[q_idx], "structural_family_key")
                    == object_family_value(objects[top_idx], "structural_family_key")
                    else 0.0,
                    "same_local_family": 1.0
                    if object_family_value(objects[q_idx], "local_family_key")
                    == object_family_value(objects[top_idx], "local_family_key")
                    else 0.0,
                    "top_structural_probability": top_score,
                    "top_local_probability": top_local_prob,
                    "best_positive_probability": best_positive,
                    "best_decoy_probability": best_decoy,
                    "learned_margin": best_positive - best_decoy,
                    "geometric_top_score": _geometric_score(objects[q_idx], objects[top_idx]),
                }
            )
    return {
        "query_count": len(rows),
        "top1_structural_accuracy": _mean([float(row["same_structural_family"]) for row in rows]),
        "top1_local_leakage": _mean([float(row["same_local_family"]) for row in rows]),
        "top_same_case_fraction": _mean([float(row["same_case"]) for row in rows]),
        "mean_top_structural_probability": _mean([float(row["top_structural_probability"]) for row in rows]),
        "mean_top_local_probability": _mean([float(row["top_local_probability"]) for row in rows]),
        "mean_learned_margin": _mean([float(row["learned_margin"]) for row in rows]),
        "mean_geometric_score_at_learned_top": _mean([float(row["geometric_top_score"]) for row in rows]),
        "rows": rows,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    holdout = payload["holdout_retrieval"]
    all_eval = payload["all_case_retrieval"]
    lines = [
        "# Learned Structural Family Probe",
        "",
        "This assay trains a tiny pairwise neural probe over RAFA law-object features to predict structural-family agreement while keeping local identity as a separate auxiliary target.",
        "",
        "## Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- object count: `{summary['object_count']}`",
        f"- train objects: `{summary['train_object_count']}`",
        f"- holdout objects: `{summary['holdout_object_count']}`",
        f"- holdout case: `{summary['holdout_case']}`",
        f"- pair count: `{summary['pair_count']}`",
        f"- final loss: `{summary['final_loss']}`",
        f"- final structural loss: `{summary['final_structural_loss']}`",
        f"- final local loss: `{summary['final_local_loss']}`",
        f"- checkpoint: `{payload.get('checkpoint_path', '')}`",
        "",
        "## Holdout Cross-Case Retrieval",
        "",
        f"- query count: `{holdout['query_count']}`",
        f"- top1 structural accuracy: `{holdout['top1_structural_accuracy']}`",
        f"- top1 local leakage: `{holdout['top1_local_leakage']}`",
        f"- top same-case fraction: `{holdout['top_same_case_fraction']}`",
        f"- mean learned margin: `{holdout['mean_learned_margin']}`",
        f"- mean top structural probability: `{holdout['mean_top_structural_probability']}`",
        f"- mean top local probability: `{holdout['mean_top_local_probability']}`",
        "",
        "## All-Case Retrieval",
        "",
        f"- query count: `{all_eval['query_count']}`",
        f"- top1 structural accuracy: `{all_eval['top1_structural_accuracy']}`",
        f"- top1 local leakage: `{all_eval['top1_local_leakage']}`",
        f"- top same-case fraction: `{all_eval['top_same_case_fraction']}`",
        f"- mean learned margin: `{all_eval['mean_learned_margin']}`",
        "",
        "## Feature Names",
        "",
    ]
    for name in payload.get("feature_names", []):
        lines.append(f"- `{name}`")
    lines.extend(["", "## Sources", ""])
    for source in payload.get("source_files", []):
        lines.append(f"- `{source}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
    inputs: list[str],
    out_dir: Path,
    synthetic_smoke: bool,
    holdout_case: str | None,
    pair_limit: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    seed: int,
    device_name: str,
) -> dict[str, str]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    objects, source_files = _load_objects(inputs, synthetic_smoke=synthetic_smoke)
    features = _feature_tensor(objects, device)
    train_idx, test_idx, holdout = _case_split(objects, holdout_case)
    pairs, y_struct, y_local = _sample_pairs(objects, train_idx, pair_limit=pair_limit, seed=seed)
    model = PairStructuralProbe(features.size(1), hidden_dim=hidden_dim).to(device)
    train_summary = _train_probe(
        model,
        features,
        pairs,
        y_struct,
        y_local,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    holdout_retrieval = _evaluate_retrieval(
        model,
        objects,
        features,
        test_idx,
        train_idx,
        batch_size=batch_size,
    )
    all_indices = list(range(len(objects)))
    all_case_retrieval = _evaluate_retrieval(
        model,
        objects,
        features,
        all_indices,
        all_indices,
        batch_size=batch_size,
    )
    summary = {
        "status": "pass_learned_structural_signal"
        if holdout_retrieval["top1_structural_accuracy"] > 0.0
        else "fail_no_learned_structural_signal",
        "object_count": len(objects),
        "train_object_count": len(train_idx),
        "holdout_object_count": len(test_idx),
        "holdout_case": holdout,
        "pair_count": len(pairs),
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
            "pair_limit": int(pair_limit),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "hidden_dim": int(hidden_dim),
            "seed": int(seed),
            "device": str(device),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "learned_structural_family_probe.json"
    md_path = out_dir / "LEARNED_STRUCTURAL_FAMILY_PROBE.md"
    checkpoint_path = out_dir / "learned_structural_family_probe.pt"
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
    ap = argparse.ArgumentParser(description="Train/evaluate a learned structural-family probe over RAFA law objects.")
    ap.add_argument("--input", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synthetic-smoke", action="store_true")
    ap.add_argument("--holdout-case", default=None)
    ap.add_argument("--pair-limit", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(
        json.dumps(
            run_probe(
                inputs=[str(item) for item in args.input],
                out_dir=Path(args.out_dir),
                synthetic_smoke=bool(args.synthetic_smoke),
                holdout_case=args.holdout_case,
                pair_limit=int(args.pair_limit),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                hidden_dim=int(args.hidden_dim),
                seed=int(args.seed),
                device_name=str(args.device),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
