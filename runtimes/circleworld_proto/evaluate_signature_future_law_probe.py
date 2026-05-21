from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import load_config  # noqa: E402
from evaluate_signature_operator_tail_probe import (  # noqa: E402
    _baseline_mse,
    _fit_linear_probe,
    _mse,
    _predict_linear_probe,
    _relative_improvement,
    _standardize_train_heldout,
)
from evaluate_signature_tail_incremental_usefulness import (  # noqa: E402
    _learned_rows_from_non_law_input,
    _train_non_law_input_model,
)
from train_learned_signature_scout import (  # noqa: E402
    DEFAULT_CONFIG,
    TOKENBURST_ROOT,
    _case_runs,
    _concat_factor_packs,
    _default_cases_for_scout,
    _factor_pack_for_run,
    _load_cases,
    _rotate_cases,
    _safe_device,
    _select_case_slice,
    _set_rng_seed,
    _shuffle_cases,
    _transform_specs,
)
from export_circleworld_audio import _load_circle_cfg  # noqa: E402


SCHEMA = "rafa_signature_future_law_probe_v0"
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "signature_future_law_probe_2026_05_07"
DEFAULT_TARGETS = (
    "future_law_signature",
    "future_operator_seed",
    "delta_law_signature",
    "delta_operator_seed",
)
DEFAULT_RIDGE_ALPHAS = (1.0e-1,)


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


def _pool_rows(rows: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
    if rows.numel() == 0:
        return torch.zeros(1, 1, device=rows.device, dtype=rows.dtype)
    pooled = rows.detach().float().mean(dim=0, keepdim=True)
    return F.normalize(pooled, dim=-1) if normalize else pooled


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


def _full_factor_features(pack: Any) -> torch.Tensor:
    return torch.cat(
        [
            _metadata_features(pack),
            pack.law_signature.detach().float(),
            pack.operator_seed.detach().float(),
        ],
        dim=-1,
    )


def _collect_packs(
    case_runs: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[list[Any], list[int], list[int], list[bool]]:
    packs: list[Any] = []
    packet_case_ids: list[int] = []
    packet_run_ids: list[int] = []
    packet_is_base: list[bool] = []
    next_run_id = 0
    for case_idx, row in enumerate(case_runs):
        base_pack = _factor_pack_for_run(row["base_run"], device)
        if base_pack is not None:
            count = int(base_pack.q_profile.size(0))
            packs.append(base_pack)
            packet_case_ids.extend([case_idx] * count)
            packet_run_ids.extend([next_run_id] * count)
            packet_is_base.extend([True] * count)
            next_run_id += 1
        for transform_run in row["transform_runs"]:
            pack_t = _factor_pack_for_run(transform_run["run"], device)
            if pack_t is None:
                continue
            count = int(pack_t.q_profile.size(0))
            packs.append(pack_t)
            packet_case_ids.extend([case_idx] * count)
            packet_run_ids.extend([next_run_id] * count)
            packet_is_base.extend([False] * count)
            next_run_id += 1
    return packs, packet_case_ids, packet_run_ids, packet_is_base


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
    rows = []
    for case_name, wav_path_str in cases.items():
        rows.append(
            _case_runs(
                case_name=case_name,
                wav_path=Path(wav_path_str),
                cfg=cfg,
                stft_cfg=stft_cfg,
                device=device,
                device_name=device_name,
                clip_seconds=int(clip_seconds),
                transforms=transforms,
            )
        )
    return rows


def _examples_from_case_runs(
    *,
    model: Any,
    case_runs: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, Any]]]:
    view_rows: dict[str, list[torch.Tensor]] = {
        "learned_base_full_h": [],
        "learned_base_operator_tail_h_640_768": [],
        "learned_base_prefix_h_0_640": [],
        "explicit_base_non_law_metadata": [],
        "explicit_base_full_factor_upper_bound": [],
    }
    target_rows: dict[str, list[torch.Tensor]] = {target: [] for target in DEFAULT_TARGETS}
    meta_rows: list[dict[str, Any]] = []
    for row in case_runs:
        base_pack = _factor_pack_for_run(row["base_run"], device)
        if base_pack is None or base_pack.q_profile.numel() == 0:
            continue
        base_h = _learned_rows_from_non_law_input(model, base_pack, device)
        base_law = _pool_rows(base_pack.law_signature, normalize=True)
        base_op = _pool_rows(base_pack.operator_seed, normalize=True)
        base_views = {
            "learned_base_full_h": _pool_rows(base_h, normalize=True),
            "learned_base_operator_tail_h_640_768": _pool_rows(base_h[:, 640:768], normalize=True),
            "learned_base_prefix_h_0_640": _pool_rows(base_h[:, :640], normalize=True),
            "explicit_base_non_law_metadata": _pool_rows(_metadata_features(base_pack), normalize=False),
            "explicit_base_full_factor_upper_bound": _pool_rows(_full_factor_features(base_pack), normalize=False),
        }
        for transform_run in row.get("transform_runs", []):
            transform_pack = _factor_pack_for_run(transform_run["run"], device)
            if transform_pack is None or transform_pack.q_profile.numel() == 0:
                continue
            future_law = _pool_rows(transform_pack.law_signature, normalize=True)
            future_op = _pool_rows(transform_pack.operator_seed, normalize=True)
            for view_name, view in base_views.items():
                view_rows[view_name].append(view.squeeze(0))
            target_rows["future_law_signature"].append(future_law.squeeze(0))
            target_rows["future_operator_seed"].append(future_op.squeeze(0))
            target_rows["delta_law_signature"].append((future_law - base_law).squeeze(0))
            target_rows["delta_operator_seed"].append((future_op - base_op).squeeze(0))
            spec = transform_run.get("spec") if isinstance(transform_run.get("spec"), dict) else {}
            meta_rows.append(
                {
                    "case": row.get("case"),
                    "transform": spec.get("name", "transform"),
                    "base_packet_count": int(base_pack.q_profile.size(0)),
                    "future_packet_count": int(transform_pack.q_profile.size(0)),
                }
            )
    views = {name: torch.stack(values, dim=0) for name, values in view_rows.items() if values}
    targets = {name: torch.stack(values, dim=0) for name, values in target_rows.items() if values}
    return views, targets, meta_rows


def _probe_one_view(
    *,
    train_x: torch.Tensor,
    heldout_x: torch.Tensor,
    train_y: torch.Tensor,
    heldout_y: torch.Tensor,
    ridge_alphas: tuple[float, ...],
) -> dict[str, Any]:
    train_z, heldout_z, standardization = _standardize_train_heldout(train_x, heldout_x)
    baseline = _baseline_mse(train_y, heldout_y)
    rows = []
    for alpha in ridge_alphas:
        weights = _fit_linear_probe(train_z, train_y, ridge_alpha=float(alpha))
        train_pred = _predict_linear_probe(train_z, weights)
        heldout_pred = _predict_linear_probe(heldout_z, weights)
        train_mse = _mse(train_pred, train_y)
        heldout_mse = _mse(heldout_pred, heldout_y)
        rows.append(
            {
                "probe": "ridge" if float(alpha) > 0.0 else "least_squares",
                "ridge_alpha": float(alpha),
                "train_mse": train_mse,
                "heldout_mse": heldout_mse,
                "baseline_heldout_mse": baseline,
                "relative_improvement_vs_train_mean": _relative_improvement(baseline, heldout_mse),
            }
        )
    rows.sort(key=lambda item: (float(item["heldout_mse"]), float(item["ridge_alpha"])))
    return {
        "feature_dim": int(train_x.size(-1)),
        "standardization": standardization,
        "baseline_heldout_mse": baseline,
        "best_by_heldout_mse": rows[0] if rows else {},
        "probes": rows,
    }


def _probe_matrix(
    *,
    train_views: dict[str, torch.Tensor],
    heldout_views: dict[str, torch.Tensor],
    train_targets: dict[str, torch.Tensor],
    heldout_targets: dict[str, torch.Tensor],
    targets: tuple[str, ...],
    ridge_alphas: tuple[float, ...],
) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for target_name in targets:
        matrix[target_name] = {
            "target_dim": int(train_targets[target_name].size(-1)),
            "views": {},
        }
        for view_name, train_x in train_views.items():
            matrix[target_name]["views"][view_name] = _probe_one_view(
                train_x=train_x,
                heldout_x=heldout_views[view_name],
                train_y=train_targets[target_name],
                heldout_y=heldout_targets[target_name],
                ridge_alphas=ridge_alphas,
            )
    return matrix


def _rank_views(matrix: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    ranked: dict[str, list[dict[str, Any]]] = {}
    for target_name, target_row in matrix.items():
        rows = []
        for view_name, view_row in target_row["views"].items():
            best = view_row["best_by_heldout_mse"]
            rows.append(
                {
                    "view": view_name,
                    "feature_dim": view_row["feature_dim"],
                    "heldout_mse": best.get("heldout_mse"),
                    "baseline_heldout_mse": best.get("baseline_heldout_mse"),
                    "relative_improvement_vs_train_mean": best.get("relative_improvement_vs_train_mean"),
                    "probe": best.get("probe"),
                    "ridge_alpha": best.get("ridge_alpha"),
                }
            )
        rows.sort(key=lambda row: (_finite_float(row.get("heldout_mse"), float("inf")), str(row.get("view"))))
        ranked[target_name] = rows
    return ranked


def _verdict(rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    target_rows = []
    learned_win_count = 0
    learned_beats_non_law_count = 0
    full_factor_win_count = 0
    for target_name, rows in rankings.items():
        best = rows[0] if rows else {}
        learned_rows = [row for row in rows if str(row.get("view", "")).startswith("learned_")]
        non_law = next((row for row in rows if row.get("view") == "explicit_base_non_law_metadata"), {})
        full_factor = next((row for row in rows if row.get("view") == "explicit_base_full_factor_upper_bound"), {})
        best_learned = min(
            learned_rows,
            key=lambda row: _finite_float(row.get("heldout_mse"), float("inf")),
            default={},
        )
        learned_wins = bool(best.get("view") in {row.get("view") for row in learned_rows})
        learned_beats_non_law = _finite_float(best_learned.get("heldout_mse"), float("inf")) < _finite_float(
            non_law.get("heldout_mse"),
            float("inf"),
        )
        full_factor_wins = best.get("view") == "explicit_base_full_factor_upper_bound"
        learned_win_count += int(learned_wins)
        learned_beats_non_law_count += int(learned_beats_non_law)
        full_factor_win_count += int(full_factor_wins)
        target_rows.append(
            {
                "target": target_name,
                "best_view": best.get("view"),
                "best_heldout_mse": best.get("heldout_mse"),
                "best_learned_view": best_learned.get("view"),
                "best_learned_heldout_mse": best_learned.get("heldout_mse"),
                "explicit_non_law_heldout_mse": non_law.get("heldout_mse"),
                "explicit_full_factor_heldout_mse": full_factor.get("heldout_mse"),
                "learned_wins": learned_wins,
                "learned_beats_non_law": learned_beats_non_law,
                "full_factor_wins": full_factor_wins,
            }
        )
    target_count = len(rankings)
    if target_count and learned_win_count == target_count:
        status = "future_law_learned_view_wins_all"
    elif target_count and learned_beats_non_law_count >= max(1, math.ceil(target_count / 2)):
        status = "future_law_learned_beats_non_law_metadata"
    elif full_factor_win_count == target_count:
        status = "future_law_explicit_full_factor_upper_bound_only"
    else:
        status = "future_law_no_learned_usefulness"
    return {
        "status": status,
        "promotion_effect": "none",
        "target_count": target_count,
        "learned_win_count": learned_win_count,
        "learned_beats_non_law_count": learned_beats_non_law_count,
        "full_factor_win_count": full_factor_win_count,
        "target_rows": target_rows,
    }


def evaluate_signature_future_law_probe(
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
    case_shuffle_seed: int | None,
    selection_objective: str,
    targets: tuple[str, ...],
    ridge_alphas: tuple[float, ...],
) -> dict[str, Any]:
    device = _safe_device(device_name)
    _set_rng_seed(int(seed))
    cfg = _load_circle_cfg(config_path)
    stft_cfg = load_config()["data"]["stft"]
    transforms = _transform_specs()[: max(1, int(num_transforms))]
    all_cases = _rotate_cases(_shuffle_cases(_load_cases(cases_path), case_shuffle_seed), int(case_offset))
    train_selected = _select_case_slice(all_cases, start=0, count=int(num_cases))
    heldout_selected = _select_case_slice(
        all_cases,
        start=len(train_selected),
        count=int(heldout_cases) if int(heldout_cases) > 0 else 0,
    )
    if not train_selected or not heldout_selected:
        raise RuntimeError("future-law probe requires non-empty train and heldout case sets")
    train_case_runs = _build_case_runs(
        cases=train_selected,
        cfg=cfg,
        stft_cfg=stft_cfg,
        device=device,
        device_name=device_name,
        clip_seconds=int(clip_seconds),
        transforms=transforms,
    )
    heldout_case_runs = _build_case_runs(
        cases=heldout_selected,
        cfg=cfg,
        stft_cfg=stft_cfg,
        device=device,
        device_name=device_name,
        clip_seconds=int(clip_seconds),
        transforms=transforms,
    )
    packs, packet_case_ids, packet_run_ids, packet_is_base = _collect_packs(train_case_runs, device=device)
    train_pack = _concat_factor_packs(packs)
    _set_rng_seed(int(seed))
    model, train_summary = _train_non_law_input_model(
        train_pack,
        packet_case_ids=torch.tensor(packet_case_ids, dtype=torch.long, device=device),
        packet_run_ids=torch.tensor(packet_run_ids, dtype=torch.long, device=device),
        packet_is_base=torch.tensor(packet_is_base, dtype=torch.bool, device=device),
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
    train_views, train_targets, train_examples = _examples_from_case_runs(
        model=model,
        case_runs=train_case_runs,
        device=device,
    )
    heldout_views, heldout_targets, heldout_examples = _examples_from_case_runs(
        model=model,
        case_runs=heldout_case_runs,
        device=device,
    )
    matrix = _probe_matrix(
        train_views=train_views,
        heldout_views=heldout_views,
        train_targets=train_targets,
        heldout_targets=heldout_targets,
        targets=targets,
        ridge_alphas=ridge_alphas,
    )
    rankings = _rank_views(matrix)
    verdict = _verdict(rankings)
    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "runtime": "circleworld_proto",
        "purpose": "bounded_future_law_usefulness_probe",
        "promotion_effect": "none",
        "interpretation": (
            "This probe tests whether base-window dense signatures predict transformed heldout law/operator surfaces. "
            "It is a redacted ranking diagnostic, not an audio continuation proof and not the primary incremental "
            "tail claim. Learned h is encoded from inputs where law_signature and operator_seed are zeroed. The "
            "explicit full-factor view remains an upper-bound control because it intentionally includes base "
            "law/operator factors; explicit non-law metadata excludes law_signature/operator_seed."
        ),
        "config_path": str(config_path),
        "cases_path": str(cases_path),
        "out_dir": str(out_dir),
        "device": str(device),
        "requested_device": str(device_name),
        "seed": int(seed),
        "case_offset": int(case_offset),
        "case_shuffle_seed": None if case_shuffle_seed is None else int(case_shuffle_seed),
        "num_cases": len(train_selected),
        "heldout_cases": len(heldout_selected),
        "num_transforms": len(transforms),
        "clip_seconds": int(clip_seconds),
        "train_case_names": list(train_selected.keys()),
        "heldout_case_names": list(heldout_selected.keys()),
        "train_example_count": len(train_examples),
        "heldout_example_count": len(heldout_examples),
        "targets": list(targets),
        "ridge_alphas": [float(alpha) for alpha in ridge_alphas],
        "ridge_alpha_selection_policy": "fixed" if len(ridge_alphas) == 1 else "heldout_mse_best_diagnostic",
        "leakage_guard": {
            "encoder_law_signature_input": "zeroed",
            "encoder_operator_seed_input": "zeroed",
            "target_future_law_signature": "real",
            "target_future_operator_seed": "real",
        },
        "train_summary": train_summary,
        "feature_views": {
            "learned_base_full_h": "pooled base-window learned h from redacted non-law encoder input",
            "learned_base_operator_tail_h_640_768": "pooled base h[:,640:768] from redacted non-law encoder input",
            "learned_base_prefix_h_0_640": "pooled base h[:640] from redacted non-law encoder input",
            "explicit_base_non_law_metadata": "base coarse/q/arc/temporal/support/branch/confidence only",
            "explicit_base_full_factor_upper_bound": "base non-law metadata plus law_signature and operator_seed",
        },
        "probe_matrix": matrix,
        "rankings": rankings,
        "verdict": verdict,
        "status": verdict["status"],
        "train_examples": train_examples,
        "heldout_examples": heldout_examples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "signature_future_law_probe.json"
    md_path = out_dir / "SIGNATURE_FUTURE_LAW_PROBE.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    verdict = summary["verdict"]
    train = summary["train_summary"]
    lines = [
        "# Signature Future Law Probe",
        "",
        f"- schema: `{summary['schema']}`",
        f"- status: `{summary['status']}`",
        f"- promotion effect: `{summary['promotion_effect']}`",
        f"- seed: `{summary['seed']}`",
        f"- case offset: `{summary['case_offset']}`",
        f"- case shuffle seed: `{summary['case_shuffle_seed']}`",
        f"- train cases: `{summary['num_cases']}`",
        f"- heldout cases: `{summary['heldout_cases']}`",
        f"- transforms: `{summary['num_transforms']}`",
        f"- train examples: `{summary['train_example_count']}`",
        f"- heldout examples: `{summary['heldout_example_count']}`",
        f"- hidden dim: `{train.get('hidden_dim')}`",
        f"- run contrastive weight: `{train.get('run_contrastive_weight')}`",
        f"- metamer consistency weight: `{train.get('metamer_consistency_weight')}`",
        f"- base SVD rank weight: `{train.get('base_svd_rank_weight')}`",
        f"- final law signature loss: `{train.get('final_law_signature_loss')}`",
        f"- final operator seed loss: `{train.get('final_operator_seed_loss')}`",
        f"- learned wins: `{verdict['learned_win_count']}` / `{verdict['target_count']}`",
        f"- learned beats non-law metadata: `{verdict['learned_beats_non_law_count']}` / `{verdict['target_count']}`",
        f"- full-factor upper-bound wins: `{verdict['full_factor_win_count']}` / `{verdict['target_count']}`",
        "",
        "## Target Verdicts",
        "",
        "| target | best view | best mse | best learned | learned mse | non-law mse | full-factor mse |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in verdict.get("target_rows", []):
        lines.append(
            "| {target} | {best} | {best_mse:.6g} | {learned} | {learned_mse:.6g} | {nonlaw:.6g} | {full:.6g} |".format(
                target=row.get("target"),
                best=row.get("best_view"),
                best_mse=_finite_float(row.get("best_heldout_mse")),
                learned=row.get("best_learned_view"),
                learned_mse=_finite_float(row.get("best_learned_heldout_mse")),
                nonlaw=_finite_float(row.get("explicit_non_law_heldout_mse")),
                full=_finite_float(row.get("explicit_full_factor_heldout_mse")),
            )
        )
    lines.extend(["", "## Rankings", ""])
    for target_name, rows in summary["rankings"].items():
        lines.extend([f"### {target_name}", ""])
        lines.append("| rank | view | alpha | heldout mse | baseline mse | rel improvement |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for idx, row in enumerate(rows, start=1):
            lines.append(
                "| {rank} | {view} | {alpha} | {mse:.6g} | {base:.6g} | {rel:.6g} |".format(
                    rank=idx,
                    view=row.get("view"),
                    alpha=row.get("ridge_alpha"),
                    mse=_finite_float(row.get("heldout_mse")),
                    base=_finite_float(row.get("baseline_heldout_mse")),
                    rel=_finite_float(row.get("relative_improvement_vs_train_mean")),
                )
            )
        lines.append("")
    lines.extend(["## Interpretation", "", summary["interpretation"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a learned RAFA signature encoder and probe future law/operator surfaces from base signatures."
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
    parser.add_argument("--case-shuffle-seed", type=int, default=None)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--ridge-alphas", default=",".join(str(value) for value in DEFAULT_RIDGE_ALPHAS))
    args = parser.parse_args()
    targets = _parse_csv_names(str(args.targets), DEFAULT_TARGETS, option_name="--targets")
    ridge_alphas = _parse_csv_floats(str(args.ridge_alphas))
    summary = evaluate_signature_future_law_probe(
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
        seed=int(args.seed),
        case_offset=int(args.case_offset),
        case_shuffle_seed=args.case_shuffle_seed,
        selection_objective=str(args.selection_objective),
        targets=targets,
        ridge_alphas=ridge_alphas,
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "saved": str(Path(args.out_dir) / "signature_future_law_probe.json"),
                "report": str(Path(args.out_dir) / "SIGNATURE_FUTURE_LAW_PROBE.md"),
                "status": summary["status"],
                "learned_win_count": summary["verdict"]["learned_win_count"],
                "learned_beats_non_law_count": summary["verdict"]["learned_beats_non_law_count"],
                "target_count": summary["verdict"]["target_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
