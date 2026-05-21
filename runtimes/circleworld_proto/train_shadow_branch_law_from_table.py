from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from circleworld import (  # noqa: E402
    RafaLearnedBranchLawV0,
    learned_branch_law_output_names,
    learned_branch_law_output_tensor,
)


SHADOW_BRANCH_FEATURE_NAMES: tuple[str, ...] = (
    "causal_selector_model_score",
    "identity_signal",
    "recurrence_compatibility",
    "jump_safety",
    "hard_decoy_against_causal",
    "causal_safety_win_over_decoy",
    "causal_parent_phase_divergence_norm",
    "decoy_parent_phase_divergence_norm",
    "causal_world_jump_proxy_norm",
    "decoy_world_jump_proxy_norm",
    "decoy_causal_phase_divergence_gap_signed",
    "decoy_causal_world_jump_gap_signed",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _norm_positive(value: float, scale: float) -> float:
    return _clamp01(float(value) / max(float(scale), 1.0e-8))


def _norm_signed(value: float, scale: float) -> float:
    return _clamp(float(value) / max(float(scale), 1.0e-8), -1.0, 1.0)


def shadow_branch_feature_names() -> tuple[str, ...]:
    return SHADOW_BRANCH_FEATURE_NAMES


def shadow_row_to_feature_target(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    causal_div = _as_float(row.get("causal_parent_phase_divergence", 0.0))
    decoy_div = _as_float(row.get("decoy_parent_phase_divergence", 0.0))
    causal_jump = _as_float(row.get("causal_world_jump_proxy", 0.0))
    decoy_jump = _as_float(row.get("decoy_world_jump_proxy", 0.0))
    phase_gap = _as_float(row.get("decoy_causal_phase_divergence_gap", decoy_div - causal_div))
    jump_gap = _as_float(row.get("decoy_causal_world_jump_gap", decoy_jump - causal_jump))

    features = [
        _clamp01(_as_float(row.get("causal_selector_model_score", 0.0))),
        _clamp01(_as_float(row.get("identity_signal", 0.0))),
        _clamp01(_as_float(row.get("recurrence_compatibility", 0.0))),
        _clamp01(_as_float(row.get("jump_safety", 0.0))),
        _clamp01(_as_float(row.get("hard_decoy_against_causal", 0.0))),
        _clamp01(_as_float(row.get("causal_safety_win_over_decoy", 0.0))),
        _norm_positive(causal_div, 0.35),
        _norm_positive(decoy_div, 0.35),
        _norm_positive(causal_jump, 0.08),
        _norm_positive(decoy_jump, 0.08),
        _norm_signed(phase_gap, 0.05),
        _norm_signed(jump_gap, 0.03),
    ]

    survival_signed = _clamp(2.0 * _as_float(row.get("survival_delta", 0.0)) - 1.0, -1.0, 1.0)
    support_delta = _clamp(_as_float(row.get("support_delta", 0.0)), -1.0, 1.0)
    coexistence = _clamp01(_as_float(row.get("coexistence_drive", 0.0)))
    split_drive = _clamp01(0.70 * coexistence + 0.30 * _as_float(row.get("hard_decoy_against_causal", 0.0)))
    targets_by_name = {
        "split_drive": split_drive,
        "coexistence_drive": coexistence,
        "merge_drive": _clamp01(_as_float(row.get("merge_drive", 0.0))),
        "survival_delta0": survival_signed,
        "survival_delta1": survival_signed,
        "collapse_pressure": _clamp01(_as_float(row.get("collapse_pressure", 0.0))),
        "support_delta0": support_delta,
        "support_delta1": support_delta,
        "coherence_target": _clamp01(_as_float(row.get("coherence_target", 0.0))),
        "qtrace_inheritance_mix": _clamp01(_as_float(row.get("q_trace_inheritance_mix", 0.0))),
    }
    targets = [float(targets_by_name[name]) for name in learned_branch_law_output_names()]
    return features, targets


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []) or []:
            clone = dict(row)
            clone["_table_path"] = str(path)
            rows.append(clone)
    if not rows:
        raise RuntimeError("No rows found in shadow branch-law table input")
    return rows


def _tensorize(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    targets: list[list[float]] = []
    for row in rows:
        feat, target = shadow_row_to_feature_target(row)
        features.append(feat)
        targets.append(target)
    return (
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def _train_model(
    train_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    hidden_dim: int,
    seed: int,
) -> tuple[RafaLearnedBranchLawV0, dict[str, float]]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    features, targets = _tensorize(train_rows, device)
    model = RafaLearnedBranchLawV0(
        input_dim=len(SHADOW_BRANCH_FEATURE_NAMES),
        hidden_dim=int(hidden_dim),
        output_dim=len(learned_branch_law_output_names()),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    losses: list[float] = []
    for _ in range(max(1, int(epochs))):
        output = learned_branch_law_output_tensor(model(features))
        loss = F.mse_loss(output, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return model, {
        "train_loss_initial": float(losses[0]),
        "train_loss_final": float(losses[-1]),
    }


def _evaluate_model(
    model: RafaLearnedBranchLawV0,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    features, targets = _tensorize(rows, device)
    names = list(learned_branch_law_output_names())
    with torch.no_grad():
        pred = learned_branch_law_output_tensor(model(features))
        diff = (pred - targets).abs()
        per_output = {
            name: {
                "mae": float(diff[:, idx].mean().detach().cpu().item()),
                "target_mean": float(targets[:, idx].mean().detach().cpu().item()),
                "prediction_mean": float(pred[:, idx].mean().detach().cpu().item()),
            }
            for idx, name in enumerate(names)
        }
    return {
        "row_count": len(rows),
        "mean_mae": float(diff.mean().detach().cpu().item()) if len(rows) else 0.0,
        "max_mae": float(diff.max().detach().cpu().item()) if len(rows) else 0.0,
        "per_output": per_output,
    }


def _saliency(
    model: RafaLearnedBranchLawV0,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, float]:
    features, targets = _tensorize(rows, device)
    features = features.detach().clone().requires_grad_(True)
    pred = learned_branch_law_output_tensor(model(features))
    loss = F.mse_loss(pred, targets)
    loss.backward()
    grad = features.grad.detach().abs().mean(dim=0).cpu().tolist()
    return {name: float(grad[idx]) for idx, name in enumerate(SHADOW_BRANCH_FEATURE_NAMES)}


def train_shadow_branch_law(
    *,
    table_paths: list[Path],
    out_dir: Path,
    device_name: str,
    epochs: int,
    lr: float,
    hidden_dim: int,
    seed: int,
) -> dict[str, str]:
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    rows = _load_rows(table_paths)
    source_keys = sorted({str(row.get("source_path", row.get("_table_path", ""))) for row in rows})
    folds: list[dict[str, Any]] = []
    for fold_idx, source_key in enumerate(source_keys):
        train_rows = [row for row in rows if str(row.get("source_path", row.get("_table_path", ""))) != source_key]
        holdout_rows = [row for row in rows if str(row.get("source_path", row.get("_table_path", ""))) == source_key]
        if not train_rows or not holdout_rows:
            continue
        model, train_summary = _train_model(
            train_rows,
            device=device,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            seed=int(seed) + fold_idx,
        )
        folds.append(
            {
                "holdout_source": source_key,
                "train_row_count": len(train_rows),
                "holdout_row_count": len(holdout_rows),
                "training": train_summary,
                "evaluation": _evaluate_model(model, holdout_rows, device=device),
            }
        )

    final_model, final_train = _train_model(
        rows,
        device=device,
        epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    final_eval = _evaluate_model(final_model, rows, device=device)
    saliency = _saliency(final_model, rows, device=device)
    holdout_maes = [float(fold["evaluation"]["mean_mae"]) for fold in folds]
    summary = {
        "status": "pass_shadow_branch_law_fit" if final_eval["mean_mae"] < 0.05 else "warn_shadow_branch_law_fit",
        "row_count": len(rows),
        "source_count": len(source_keys),
        "fold_count": len(folds),
        "device": str(device),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "final_train_loss": float(final_train["train_loss_final"]),
        "final_mean_mae": float(final_eval["mean_mae"]),
        "final_max_mae": float(final_eval["max_mae"]),
        "mean_holdout_mae": float(np.mean(holdout_maes)) if holdout_maes else 0.0,
        "max_holdout_mae": float(np.max(holdout_maes)) if holdout_maes else 0.0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "shadow_rafa_learned_branch_law_v0.pt"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "feature_names": list(SHADOW_BRANCH_FEATURE_NAMES),
            "output_names": list(learned_branch_law_output_names()),
            "input_dim": len(SHADOW_BRANCH_FEATURE_NAMES),
            "hidden_dim": int(hidden_dim),
            "output_dim": len(learned_branch_law_output_names()),
            "summary": summary,
        },
        checkpoint_path,
    )
    payload = {
        "schema": "shadow_rafa_learned_branch_law_v0",
        "table_paths": [str(path) for path in table_paths],
        "feature_names": list(SHADOW_BRANCH_FEATURE_NAMES),
        "output_names": list(learned_branch_law_output_names()),
        "summary": summary,
        "final_training": final_train,
        "final_evaluation": final_eval,
        "feature_saliency": saliency,
        "folds": folds,
        "checkpoint_path": str(checkpoint_path),
    }
    json_path = out_dir / "shadow_branch_law_training.json"
    md_path = out_dir / "SHADOW_BRANCH_LAW_TRAINING.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    return {"json": str(json_path), "markdown": str(md_path), "checkpoint": str(checkpoint_path), "status": summary["status"]}


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    saliency = payload.get("feature_saliency", {})
    top_saliency = sorted(saliency.items(), key=lambda item: float(item[1]), reverse=True)[:8]
    lines = [
        "# Shadow Learned Branch Law Training",
        "",
        "This trains `RafaLearnedBranchLawV0` on assay-only shadow branch-law rows. It does not change Circleworld runtime behavior.",
        "",
        "## Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- rows: `{summary['row_count']}`",
        f"- sources: `{summary['source_count']}`",
        f"- folds: `{summary['fold_count']}`",
        f"- final train loss: `{summary['final_train_loss']}`",
        f"- final mean MAE: `{summary['final_mean_mae']}`",
        f"- final max MAE: `{summary['final_max_mae']}`",
        f"- mean holdout MAE: `{summary['mean_holdout_mae']}`",
        f"- max holdout MAE: `{summary['max_holdout_mae']}`",
        f"- checkpoint: `{payload.get('checkpoint_path', '')}`",
        "",
        "## Top Feature Saliency",
        "",
    ]
    for name, value in top_saliency:
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(
        [
            "",
            "## Leave-Source-Out Folds",
            "",
            "| holdout source | rows | mean MAE | max MAE |",
            "|---|---:|---:|---:|",
        ]
    )
    for fold in payload.get("folds", []):
        evaluation = fold["evaluation"]
        lines.append(
            "| {source} | {rows} | {mean_mae} | {max_mae} |".format(
                source=fold["holdout_source"],
                rows=evaluation["row_count"],
                mean_mae=evaluation["mean_mae"],
                max_mae=evaluation["max_mae"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a tiny assay-only RAFA learned branch law from shadow law tables.")
    ap.add_argument("--table", action="append", required=True, help="Path to shadow_branch_law_table.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=1600)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1515)
    args = ap.parse_args()
    result = train_shadow_branch_law(
        table_paths=[Path(raw) for raw in args.table],
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        epochs=int(args.epochs),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
