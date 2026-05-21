from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import load_config  # noqa: E402
from export_circleworld_audio import _load_circle_cfg  # noqa: E402
from train_learned_signature_scout import (  # noqa: E402
    DEFAULT_CONFIG,
    _case_runs,
    _concat_factor_packs,
    _default_cases_for_scout,
    _factor_pack_for_run,
    _learned_rows,
    _load_cases,
    _rotate_cases,
    _safe_device,
    _select_case_slice,
    _set_rng_seed,
    _train_model,
    _transform_specs,
)


SCHEMA = "rafa_signature_operator_tail_probe_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "signature_operator_tail_probe_2026_05_07"
DEFAULT_TARGETS = ("operator_seed", "law_signature")
DEFAULT_RIDGE_ALPHAS = (0.0, 1.0e-3, 1.0e-2, 1.0e-1)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _parse_csv_names(raw: str, allowed: tuple[str, ...], *, option_name: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError(f"{option_name} must include at least one value")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"{option_name} has unsupported values {unknown}; allowed: {list(allowed)}")
    return names


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    values = tuple(_finite_float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("--ridge-alphas must include at least one numeric value")
    if any(value < 0.0 for value in values):
        raise ValueError("--ridge-alphas must be non-negative")
    return values


def _run_rows(
    *,
    case_name: str,
    wav_path_str: str,
    cfg: Any,
    stft_cfg: dict[str, Any],
    device: torch.device,
    device_name: str,
    clip_seconds: int,
    transforms: list[dict[str, Any]],
) -> dict[str, Any]:
    return _case_runs(
        case_name=case_name,
        wav_path=Path(wav_path_str),
        cfg=cfg,
        stft_cfg=stft_cfg,
        device=device,
        device_name=device_name,
        clip_seconds=int(clip_seconds),
        transforms=transforms,
    )


def _collect_packs(
    case_runs: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[list[Any], list[int], list[int], list[bool], list[str]]:
    packs: list[Any] = []
    packet_case_ids: list[int] = []
    packet_run_ids: list[int] = []
    packet_is_base: list[bool] = []
    run_labels: list[str] = []
    next_run_id = 0
    for case_idx, row in enumerate(case_runs):
        case_name = str(row["case"])
        base_pack = _factor_pack_for_run(row["base_run"], device)
        if base_pack is not None:
            count = int(base_pack.q_profile.size(0))
            packs.append(base_pack)
            packet_case_ids.extend([case_idx] * count)
            packet_run_ids.extend([next_run_id] * count)
            packet_is_base.extend([True] * count)
            run_labels.append(f"{case_name}:base")
            next_run_id += 1
        for transform_run in row["transform_runs"]:
            transform = transform_run["spec"]
            pack_t = _factor_pack_for_run(transform_run["run"], device)
            if pack_t is None:
                continue
            count = int(pack_t.q_profile.size(0))
            packs.append(pack_t)
            packet_case_ids.extend([case_idx] * count)
            packet_run_ids.extend([next_run_id] * count)
            packet_is_base.extend([False] * count)
            run_labels.append(f"{case_name}:{transform.get('name', 'transform')}")
            next_run_id += 1
    return packs, packet_case_ids, packet_run_ids, packet_is_base, run_labels


def _metadata_features(pack: Any) -> torch.Tensor:
    branch = pack.branch_profile.reshape(pack.branch_profile.size(0), -1)
    parts = [
        pack.coarse_profile,
        pack.q_profile,
        pack.arc_profile,
        pack.temporal_profile,
        pack.support_profile,
        branch,
        pack.confidence,
    ]
    return torch.cat([part.detach().float() for part in parts], dim=-1)


def _feature_views(h: torch.Tensor, pack: Any) -> dict[str, torch.Tensor]:
    if h.dim() != 2 or int(h.size(-1)) < 768:
        raise ValueError(f"learned h must have shape [N, >=768], got {tuple(h.shape)}")
    return {
        "learned_full_h": h.detach().float(),
        "learned_operator_tail_h_640_768": h[:, 640:768].detach().float(),
        "learned_prefix_h_0_640": h[:, :640].detach().float(),
        "explicit_non_law_metadata": _metadata_features(pack),
    }


def _target_views(pack: Any) -> dict[str, torch.Tensor]:
    return {
        "operator_seed": pack.operator_seed.detach().float(),
        "law_signature": pack.law_signature.detach().float(),
    }


def _standardize_train_heldout(
    train_x: torch.Tensor,
    heldout_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    train_z = (train_x - mean) / std
    heldout_z = (heldout_x - mean) / std
    return train_z, heldout_z, {
        "mean_abs": float(mean.abs().mean().detach().cpu().item()),
        "std_mean": float(std.mean().detach().cpu().item()),
        "std_min": float(std.min().detach().cpu().item()),
    }


def _with_intercept(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.float(), torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)], dim=-1)


def _fit_linear_probe(x: torch.Tensor, y: torch.Tensor, ridge_alpha: float) -> torch.Tensor:
    x_aug = _with_intercept(x)
    y = y.float()
    alpha = float(ridge_alpha)
    if alpha <= 0.0:
        try:
            weights = torch.linalg.lstsq(x_aug, y).solution
        except RuntimeError:
            weights = torch.linalg.pinv(x_aug).matmul(y)
        if torch.isfinite(weights).all():
            return weights
        return torch.linalg.pinv(x_aug).matmul(y)
    xtx = x_aug.t().matmul(x_aug)
    penalty = torch.eye(xtx.size(0), device=x_aug.device, dtype=x_aug.dtype)
    penalty[-1, -1] = 0.0
    rhs = x_aug.t().matmul(y)
    system = xtx + alpha * penalty
    try:
        weights = torch.linalg.solve(system, rhs)
    except RuntimeError:
        weights = torch.linalg.pinv(system).matmul(rhs)
    if torch.isfinite(weights).all():
        return weights
    return torch.linalg.pinv(system).matmul(rhs)


def _predict_linear_probe(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return _with_intercept(x).matmul(weights)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() == 0 or target.numel() == 0:
        return 0.0
    value = float((pred.float() - target.float()).pow(2).mean().detach().cpu().item())
    return value if math.isfinite(value) else float("inf")


def _baseline_mse(train_y: torch.Tensor, eval_y: torch.Tensor) -> float:
    mean_y = train_y.float().mean(dim=0, keepdim=True)
    return _mse(mean_y.expand_as(eval_y), eval_y.float())


def _relative_improvement(baseline_mse: float, mse: float) -> float:
    denom = max(float(baseline_mse), 1.0e-12)
    return float((float(baseline_mse) - float(mse)) / denom)


def _probe_one_view(
    *,
    train_x: torch.Tensor,
    heldout_x: torch.Tensor,
    train_y: torch.Tensor,
    heldout_y: torch.Tensor,
    ridge_alphas: tuple[float, ...],
) -> dict[str, Any]:
    train_z, heldout_z, standardization = _standardize_train_heldout(train_x, heldout_x)
    train_baseline_mse = _baseline_mse(train_y, train_y)
    heldout_baseline_mse = _baseline_mse(train_y, heldout_y)
    rows: list[dict[str, Any]] = []
    for alpha in ridge_alphas:
        weights = _fit_linear_probe(train_z, train_y, float(alpha))
        pred_train = _predict_linear_probe(train_z, weights)
        pred_heldout = _predict_linear_probe(heldout_z, weights)
        train_mse = _mse(pred_train, train_y)
        heldout_mse = _mse(pred_heldout, heldout_y)
        rows.append(
            {
                "probe": "least_squares" if float(alpha) == 0.0 else "ridge",
                "ridge_alpha": float(alpha),
                "train_mse": train_mse,
                "heldout_mse": heldout_mse,
                "train_baseline_mse": train_baseline_mse,
                "heldout_baseline_mse": heldout_baseline_mse,
                "train_relative_improvement": _relative_improvement(train_baseline_mse, train_mse),
                "heldout_relative_improvement": _relative_improvement(heldout_baseline_mse, heldout_mse),
                "weight_fro_norm": float(weights[:-1].float().norm().detach().cpu().item()),
                "intercept_norm": float(weights[-1].float().norm().detach().cpu().item()),
            }
        )
    finite_rows = [row for row in rows if math.isfinite(float(row["heldout_mse"]))]
    best = min(finite_rows or rows, key=lambda row: (float(row["heldout_mse"]), float(row["ridge_alpha"])))
    return {
        "feature_dim": int(train_x.size(-1)),
        "train_rows": int(train_x.size(0)),
        "heldout_rows": int(heldout_x.size(0)),
        "standardization": standardization,
        "train_baseline_mse": train_baseline_mse,
        "heldout_baseline_mse": heldout_baseline_mse,
        "probes": rows,
        "best_by_heldout_mse": dict(best),
    }


def _run_probe_matrix(
    *,
    train_h: torch.Tensor,
    heldout_h: torch.Tensor,
    train_pack: Any,
    heldout_pack: Any,
    targets: tuple[str, ...],
    ridge_alphas: tuple[float, ...],
) -> dict[str, Any]:
    train_features = _feature_views(train_h, train_pack)
    heldout_features = _feature_views(heldout_h, heldout_pack)
    train_targets = _target_views(train_pack)
    heldout_targets = _target_views(heldout_pack)
    matrix: dict[str, Any] = {}
    for target_name in targets:
        matrix[target_name] = {
            "target_dim": int(train_targets[target_name].size(-1)),
            "views": {},
        }
        for view_name, train_x in train_features.items():
            matrix[target_name]["views"][view_name] = _probe_one_view(
                train_x=train_x,
                heldout_x=heldout_features[view_name],
                train_y=train_targets[target_name],
                heldout_y=heldout_targets[target_name],
                ridge_alphas=ridge_alphas,
            )
    return matrix


def _rank_views(matrix: dict[str, Any]) -> dict[str, Any]:
    ranked: dict[str, Any] = {}
    for target_name, target_row in matrix.items():
        rows = []
        for view_name, view_row in target_row["views"].items():
            best = view_row["best_by_heldout_mse"]
            rows.append(
                {
                    "view": view_name,
                    "best_probe": best["probe"],
                    "ridge_alpha": best["ridge_alpha"],
                    "heldout_mse": best["heldout_mse"],
                    "heldout_baseline_mse": best["heldout_baseline_mse"],
                    "heldout_relative_improvement": best["heldout_relative_improvement"],
                    "feature_dim": view_row["feature_dim"],
                }
            )
        rows.sort(key=lambda row: (float(row["heldout_mse"]), str(row["view"])))
        ranked[target_name] = rows
    return ranked


def _build_case_runs(
    *,
    cases: dict[str, str],
    cfg: Any,
    stft_cfg: dict[str, Any],
    device: torch.device,
    device_name: str,
    clip_seconds: int,
    transforms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name, wav_path_str in cases.items():
        rows.append(
            _run_rows(
                case_name=case_name,
                wav_path_str=wav_path_str,
                cfg=cfg,
                stft_cfg=stft_cfg,
                device=device,
                device_name=device_name,
                clip_seconds=int(clip_seconds),
                transforms=transforms,
            )
        )
    return rows


def evaluate_signature_operator_tail_probe(
    *,
    config_path: Path,
    cases_path: Path,
    out_dir: Path,
    device_name: str,
    num_cases: int,
    heldout_cases: int,
    num_transforms: int,
    clip_seconds: int,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    batch_size: int,
    metamer_consistency_weight: float,
    case_separation_weight: float,
    separation_margin: float,
    run_contrastive_weight: float,
    run_contrastive_temperature: float,
    rank_entropy_weight: float,
    rank_entropy_floor: float,
    run_rank_entropy_weight: float,
    run_rank_effective_dim_floor: float,
    base_rank_entropy_weight: float,
    base_rank_effective_dim_floor: float,
    base_svd_rank_weight: float,
    base_svd_rank_floor: float,
    factor_geometry_weight: float,
    seed: int,
    case_offset: int,
    targets: tuple[str, ...],
    ridge_alphas: tuple[float, ...],
    selection_objective: str,
) -> dict[str, Any]:
    """Train a learned signature lane and probe factor-pack surfaces.

    This is a bounded usefulness probe. Targets are Circleworld factor-pack
    law/operator surfaces (`law_signature`, `operator_seed`), not an audio
    continuation proof and not a promotion gate.
    """

    device = _safe_device(device_name)
    _set_rng_seed(int(seed))
    cfg = _load_circle_cfg(config_path)
    stft_cfg = load_config()["data"]["stft"]
    all_cases = _rotate_cases(_load_cases(cases_path), int(case_offset))
    train_case_count = int(num_cases) if int(num_cases) > 0 else len(all_cases)
    selected_cases = _select_case_slice(all_cases, start=0, count=train_case_count)
    heldout_selected_cases = _select_case_slice(
        all_cases,
        start=len(selected_cases),
        count=int(heldout_cases) if int(heldout_cases) > 0 else 0,
    )
    if not selected_cases:
        raise ValueError("no train cases selected")
    if not heldout_selected_cases:
        raise ValueError("this probe requires at least one heldout case")

    transforms = _transform_specs()[: max(0, int(num_transforms))]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_case_runs = _build_case_runs(
        cases=selected_cases,
        cfg=cfg,
        stft_cfg=stft_cfg,
        device=device,
        device_name=device_name,
        clip_seconds=int(clip_seconds),
        transforms=transforms,
    )
    heldout_case_runs = _build_case_runs(
        cases=heldout_selected_cases,
        cfg=cfg,
        stft_cfg=stft_cfg,
        device=device,
        device_name=device_name,
        clip_seconds=int(clip_seconds),
        transforms=transforms,
    )

    train_packs, packet_case_ids, packet_run_ids, packet_is_base, train_run_labels = _collect_packs(
        train_case_runs,
        device=device,
    )
    heldout_packs, _heldout_case_ids, _heldout_run_ids, _heldout_is_base, heldout_run_labels = _collect_packs(
        heldout_case_runs,
        device=device,
    )
    train_pack = _concat_factor_packs(train_packs)
    heldout_pack = _concat_factor_packs(heldout_packs)
    packet_case_tensor = torch.tensor(packet_case_ids, dtype=torch.long, device=device)
    packet_run_tensor = torch.tensor(packet_run_ids, dtype=torch.long, device=device)
    packet_base_tensor = torch.tensor(packet_is_base, dtype=torch.bool, device=device)

    _set_rng_seed(int(seed))
    model, train_summary = _train_model(
        train_pack,
        packet_case_ids=packet_case_tensor,
        packet_run_ids=packet_run_tensor,
        packet_is_base=packet_base_tensor,
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        metamer_consistency_weight=float(metamer_consistency_weight),
        case_separation_weight=float(case_separation_weight),
        separation_margin=float(separation_margin),
        run_contrastive_weight=float(run_contrastive_weight),
        run_contrastive_temperature=float(run_contrastive_temperature),
        rank_entropy_weight=float(rank_entropy_weight),
        rank_entropy_floor=float(rank_entropy_floor),
        run_rank_entropy_weight=float(run_rank_entropy_weight),
        run_rank_effective_dim_floor=float(run_rank_effective_dim_floor),
        base_rank_entropy_weight=float(base_rank_entropy_weight),
        base_rank_effective_dim_floor=float(base_rank_effective_dim_floor),
        base_svd_rank_weight=float(base_svd_rank_weight),
        base_svd_rank_floor=float(base_svd_rank_floor),
        factor_geometry_weight=float(factor_geometry_weight),
        selection_objective=str(selection_objective),
    )
    train_summary["seed"] = int(seed)

    train_h = _learned_rows(model, train_pack, device)
    heldout_h = _learned_rows(model, heldout_pack, device)
    probe_matrix = _run_probe_matrix(
        train_h=train_h,
        heldout_h=heldout_h,
        train_pack=train_pack,
        heldout_pack=heldout_pack,
        targets=targets,
        ridge_alphas=ridge_alphas,
    )
    rankings = _rank_views(probe_matrix)
    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "runtime": "circleworld_proto",
        "purpose": "bounded_operator_tail_predictive_probe",
        "promotion_effect": "none",
        "interpretation": (
            "Targets are factor-pack law/operator surfaces, not an audio continuation proof. "
            "This isolated probe compares whether learned full h, h[:,640:768], h[:640], "
            "or explicit non-law metadata factors linearly predict heldout packet targets."
        ),
        "config_path": str(config_path),
        "cases_path": str(cases_path),
        "out_dir": str(out_dir),
        "device": str(device),
        "requested_device": str(device_name),
        "seed": int(seed),
        "case_offset": int(case_offset),
        "train_case_count": len(selected_cases),
        "heldout_case_count": len(heldout_selected_cases),
        "num_cases": len(selected_cases),
        "heldout_cases": len(heldout_selected_cases),
        "num_transforms": len(transforms),
        "clip_seconds": int(clip_seconds),
        "train_case_names": list(selected_cases.keys()),
        "heldout_case_names": list(heldout_selected_cases.keys()),
        "train_run_count": len(train_run_labels),
        "heldout_run_count": len(heldout_run_labels),
        "train_run_labels": train_run_labels,
        "heldout_run_labels": heldout_run_labels,
        "train_packet_count": int(train_pack.q_profile.size(0)),
        "heldout_packet_count": int(heldout_pack.q_profile.size(0)),
        "targets": list(targets),
        "ridge_alphas": [float(alpha) for alpha in ridge_alphas],
        "feature_views": {
            "learned_full_h": "all 768 learned signature dimensions",
            "learned_operator_tail_h_640_768": "learned h[:,640:768] operator-tail lane",
            "learned_prefix_h_0_640": "learned h[:,:640] prefix excluding the operator tail",
            "explicit_non_law_metadata": (
                "coarse/q/arc/temporal/support/branch/confidence factors; excludes law_signature and operator_seed"
            ),
        },
        "train_summary": train_summary,
        "probe_matrix": probe_matrix,
        "rankings": rankings,
    }
    json_path = out_dir / "signature_operator_tail_probe.json"
    md_path = out_dir / "SIGNATURE_OPERATOR_TAIL_PROBE.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    train = summary["train_summary"]
    lines = [
        "# Signature Operator Tail Probe",
        "",
        f"- schema: `{summary['schema']}`",
        f"- purpose: `{summary['purpose']}`",
        f"- promotion effect: `{summary['promotion_effect']}`",
        f"- device: `{summary['device']}`",
        f"- seed: `{summary['seed']}`",
        f"- case offset: `{summary['case_offset']}`",
        f"- train cases: `{summary['train_case_count']}`",
        f"- heldout cases: `{summary['heldout_case_count']}`",
        f"- transforms: `{summary['num_transforms']}`",
        f"- train packets: `{summary['train_packet_count']}`",
        f"- heldout packets: `{summary['heldout_packet_count']}`",
        f"- hidden dim: `{train['hidden_dim']}`",
        f"- epochs: `{train['epochs']}`",
        f"- learning rate: `{train['learning_rate']}`",
        f"- run contrastive weight: `{train['run_contrastive_weight']}`",
        f"- run contrastive temperature: `{train['run_contrastive_temperature']}`",
        f"- base SVD rank weight: `{train['base_svd_rank_weight']}`",
        f"- base SVD rank floor: `{train['base_svd_rank_floor']}`",
        "",
        "## Interpretation Guardrail",
        "",
        summary["interpretation"],
        "",
        "## Best Heldout Probes",
        "",
    ]
    for target_name, rows in summary["rankings"].items():
        lines.extend([f"### {target_name}", ""])
        lines.append("| rank | view | probe | alpha | heldout MSE | baseline MSE | rel improvement |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
        for rank, row in enumerate(rows, start=1):
            lines.append(
                "| {rank} | {view} | {probe} | {alpha:.6g} | {mse:.8f} | {base:.8f} | {rel:+.6f} |".format(
                    rank=rank,
                    view=row["view"],
                    probe=row["best_probe"],
                    alpha=float(row["ridge_alpha"]),
                    mse=float(row["heldout_mse"]),
                    base=float(row["heldout_baseline_mse"]),
                    rel=float(row["heldout_relative_improvement"]),
                )
            )
        lines.append("")
    lines.extend(["## Train Cases", ""])
    lines.extend([f"- {name}" for name in summary.get("train_case_names", [])])
    lines.extend(["", "## Heldout Cases", ""])
    lines.extend([f"- {name}" for name in summary.get("heldout_case_names", [])])
    lines.extend(["", "## Feature Views", ""])
    for name, description in summary.get("feature_views", {}).items():
        lines.append(f"- `{name}`: {description}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a learned RAFA signature encoder and compare bounded linear probes for heldout "
            "operator/law factor surfaces."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cases-json", default=str(_default_cases_for_scout()))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-cases", type=int, default=12)
    parser.add_argument("--heldout-cases", type=int, default=6)
    parser.add_argument("--num-transforms", type=int, default=6)
    parser.add_argument("--clip-seconds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--metamer-consistency-weight", type=float, default=0.0)
    parser.add_argument("--case-separation-weight", type=float, default=0.0)
    parser.add_argument("--separation-margin", type=float, default=0.65)
    parser.add_argument("--run-contrastive-weight", type=float, default=1.25)
    parser.add_argument("--run-contrastive-temperature", type=float, default=0.12)
    parser.add_argument("--rank-entropy-weight", type=float, default=0.0)
    parser.add_argument("--rank-entropy-floor", type=float, default=0.02)
    parser.add_argument("--run-rank-entropy-weight", type=float, default=0.0)
    parser.add_argument("--run-rank-effective-dim-floor", type=float, default=3.0)
    parser.add_argument("--base-rank-entropy-weight", type=float, default=0.0)
    parser.add_argument("--base-rank-effective-dim-floor", type=float, default=3.0)
    parser.add_argument("--base-svd-rank-weight", type=float, default=0.12)
    parser.add_argument("--base-svd-rank-floor", type=float, default=2.5)
    parser.add_argument("--factor-geometry-weight", type=float, default=0.0)
    parser.add_argument("--selection-objective", choices=("final", "train_proxy"), default="final")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--ridge-alphas", default=",".join(str(value) for value in DEFAULT_RIDGE_ALPHAS))
    args = parser.parse_args()

    targets = _parse_csv_names(str(args.targets), DEFAULT_TARGETS, option_name="--targets")
    ridge_alphas = _parse_csv_floats(str(args.ridge_alphas))
    summary = evaluate_signature_operator_tail_probe(
        config_path=Path(args.config),
        cases_path=Path(args.cases_json),
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        num_cases=int(args.num_cases),
        heldout_cases=int(args.heldout_cases),
        num_transforms=int(args.num_transforms),
        clip_seconds=int(args.clip_seconds),
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        metamer_consistency_weight=float(args.metamer_consistency_weight),
        case_separation_weight=float(args.case_separation_weight),
        separation_margin=float(args.separation_margin),
        run_contrastive_weight=float(args.run_contrastive_weight),
        run_contrastive_temperature=float(args.run_contrastive_temperature),
        rank_entropy_weight=float(args.rank_entropy_weight),
        rank_entropy_floor=float(args.rank_entropy_floor),
        run_rank_entropy_weight=float(args.run_rank_entropy_weight),
        run_rank_effective_dim_floor=float(args.run_rank_effective_dim_floor),
        base_rank_entropy_weight=float(args.base_rank_entropy_weight),
        base_rank_effective_dim_floor=float(args.base_rank_effective_dim_floor),
        base_svd_rank_weight=float(args.base_svd_rank_weight),
        base_svd_rank_floor=float(args.base_svd_rank_floor),
        factor_geometry_weight=float(args.factor_geometry_weight),
        selection_objective=str(args.selection_objective),
        seed=int(args.seed),
        case_offset=int(args.case_offset),
        targets=targets,
        ridge_alphas=ridge_alphas,
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "saved": str(Path(args.out_dir) / "signature_operator_tail_probe.json"),
                "report": str(Path(args.out_dir) / "SIGNATURE_OPERATOR_TAIL_PROBE.md"),
                "promotion_effect": summary["promotion_effect"],
                "seed": summary["seed"],
                "case_offset": summary["case_offset"],
                "train_case_count": summary["train_case_count"],
                "heldout_case_count": summary["heldout_case_count"],
                "rankings": summary["rankings"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
