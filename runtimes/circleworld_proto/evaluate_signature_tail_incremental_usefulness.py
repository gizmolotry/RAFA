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
from evaluate_signature_operator_tail_probe import (  # noqa: E402
    _baseline_mse,
    _build_case_runs,
    _collect_packs,
    _fit_linear_probe,
    _metadata_features,
    _mse,
    _predict_linear_probe,
    _relative_improvement,
    _standardize_train_heldout,
    _target_views,
)
from export_circleworld_audio import _load_circle_cfg  # noqa: E402
from rafa_relational_signature import RelationalFactorPack  # noqa: E402
from rafa_relational_signature_learning import (  # noqa: E402
    RafaRelationalSignatureEncoderV0,
    factor_pack_to_learning_input,
    matryoshka_signature_losses,
    summarize_learning_contract,
)
from train_learned_signature_scout import (  # noqa: E402
    DEFAULT_CONFIG,
    TOKENBURST_ROOT,
    _concat_factor_packs,
    _default_cases_for_scout,
    _effective_dim_floor_losses,
    _factor_geometry_losses,
    _learned_rows,
    _load_cases,
    _metamer_case_losses,
    _rank_entropy_losses,
    _rotate_cases,
    _run_center_contrastive_losses,
    _run_center_rank_entropy_losses,
    _safe_device,
    _select_case_slice,
    _set_rng_seed,
    _shuffle_cases,
    _svd_effective_rank_losses,
    _train_model,
    _train_proxy_selection_score,
    _transform_specs,
    _weighted_training_objective,
)


SCHEMA = "rafa_signature_tail_incremental_usefulness_v0"
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "signature_tail_incremental_usefulness_2026_05_07"
DEFAULT_TARGETS = ("operator_seed", "law_signature")
DEFAULT_RIDGE_ALPHAS = (1.0e-1,)
EPS = 1.0e-9


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _scalar(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        try:
            return _finite_float(value.detach().cpu().item(), default)
        except RuntimeError:
            return float(default)
    return _finite_float(value, default)


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


def _cat_feature(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cat([left.detach().float(), right.detach().float()], dim=-1)


def _permuted_rows(x: torch.Tensor, *, seed: int) -> torch.Tensor:
    if x.size(0) <= 1:
        return x.detach().float().clone()
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    order = torch.randperm(x.size(0), generator=generator, device=x.device)
    return x[order].detach().float()


def _cross_case_rows(x: torch.Tensor, case_ids: torch.Tensor) -> torch.Tensor:
    if x.size(0) <= 1 or case_ids.numel() != x.size(0):
        return _permuted_rows(x, seed=9173)
    out = x.detach().float().clone()
    unique_cases = [int(item) for item in torch.unique(case_ids.detach().cpu()).tolist()]
    for idx in range(x.size(0)):
        this_case = int(case_ids[idx].detach().cpu().item())
        candidates = torch.nonzero(case_ids != this_case, as_tuple=False).reshape(-1)
        if candidates.numel() == 0:
            out[idx] = x[(idx + 1) % x.size(0)]
        else:
            out[idx] = x[candidates[idx % candidates.numel()]]
    if len(unique_cases) <= 1:
        return _permuted_rows(x, seed=9173)
    return out


def _random_rows_like(x: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


def _redact_law_operator_inputs(pack: RelationalFactorPack) -> RelationalFactorPack:
    """Remove the target law/operator channels from encoder inputs for leakage-safe probes."""
    return RelationalFactorPack(
        coarse_profile=pack.coarse_profile.detach().float(),
        q_profile=pack.q_profile.detach().float(),
        arc_profile=pack.arc_profile.detach().float(),
        temporal_profile=pack.temporal_profile.detach().float(),
        support_profile=pack.support_profile.detach().float(),
        branch_profile=pack.branch_profile.detach().float(),
        law_signature=torch.zeros_like(pack.law_signature.detach().float()),
        confidence=pack.confidence.detach().float(),
        operator_seed=torch.zeros_like(pack.operator_seed.detach().float()),
    ).validate()


def _learned_rows_from_non_law_input(
    model: RafaRelationalSignatureEncoderV0,
    pack: RelationalFactorPack,
    device: torch.device,
) -> torch.Tensor:
    if pack is None or pack.q_profile.numel() == 0:
        return torch.zeros(0, int(model.cfg.signature_dim), device=device)
    with torch.no_grad():
        outputs = model(_redact_law_operator_inputs(pack))
    return outputs.h.detach().float()


def _train_non_law_input_model(
    train_pack: RelationalFactorPack,
    *,
    packet_case_ids: torch.Tensor,
    packet_run_ids: torch.Tensor,
    packet_is_base: torch.Tensor,
    hidden_dim: int,
    epochs: int,
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
    selection_objective: str,
) -> tuple[RafaRelationalSignatureEncoderV0, dict[str, Any]]:
    device = train_pack.q_profile.device
    selection_mode = str(selection_objective)
    if selection_mode not in {"final", "train_proxy"}:
        raise ValueError(f"unsupported selection objective: {selection_objective!r}")
    input_pack = _redact_law_operator_inputs(train_pack)
    model = RafaRelationalSignatureEncoderV0.from_factor_pack(
        input_pack,
        hidden_dim=int(hidden_dim),
        redact_law_operator_input=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    factor_rows = factor_pack_to_learning_input(input_pack, model.cfg)
    count = int(factor_rows.size(0))
    packet_case_ids = packet_case_ids.to(device=device, dtype=torch.long)
    packet_run_ids = packet_run_ids.to(device=device, dtype=torch.long)
    packet_is_base = packet_is_base.to(device=device, dtype=torch.bool)
    epoch_count = max(1, int(epochs))
    losses_by_epoch: list[float] = []
    initial_loss: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    selected_epoch = epoch_count
    selected_score: float | None = None
    final_epoch_score = 0.0
    for epoch in range(epoch_count):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_pack)
        losses = matryoshka_signature_losses(outputs, train_pack)
        aux_losses = _metamer_case_losses(
            outputs.h,
            packet_case_ids=packet_case_ids,
            separation_margin=float(separation_margin),
        )
        run_losses = _run_center_contrastive_losses(
            outputs.h,
            packet_case_ids=packet_case_ids,
            packet_run_ids=packet_run_ids,
            temperature=float(run_contrastive_temperature),
        )
        rank_losses = _rank_entropy_losses(outputs.h, entropy_floor=float(rank_entropy_floor))
        run_rank_losses = _run_center_rank_entropy_losses(
            outputs.h,
            packet_run_ids=packet_run_ids,
            effective_dim_floor=float(run_rank_effective_dim_floor),
        )
        base_rank_losses = _effective_dim_floor_losses(
            outputs.h[packet_is_base],
            effective_dim_floor=float(base_rank_effective_dim_floor),
            prefix="base_rank",
        )
        base_svd_losses = _svd_effective_rank_losses(
            outputs.h[packet_is_base],
            effective_rank_floor=float(base_svd_rank_floor),
            prefix="base_svd_rank",
        )
        factor_geometry_losses = _factor_geometry_losses(outputs.h, factor_rows)
        loss = _weighted_training_objective(
            losses,
            aux_losses,
            run_losses,
            rank_losses,
            run_rank_losses,
            base_rank_losses,
            base_svd_losses,
            metamer_consistency_weight=metamer_consistency_weight,
            case_separation_weight=case_separation_weight,
            run_contrastive_weight=run_contrastive_weight,
            rank_entropy_weight=rank_entropy_weight,
            run_rank_entropy_weight=run_rank_entropy_weight,
            base_rank_entropy_weight=base_rank_entropy_weight,
            base_svd_rank_weight=base_svd_rank_weight,
        )
        loss = loss + float(factor_geometry_weight) * factor_geometry_losses["factor_geometry_loss"]
        loss.backward()
        optimizer.step()
        loss_value = _scalar(loss)
        if initial_loss is None:
            initial_loss = loss_value
        losses_by_epoch.append(loss_value)
        final_epoch_score = _train_proxy_selection_score(
            aux_losses,
            run_losses,
            base_rank_losses,
            base_svd_losses,
            factor_geometry_losses,
            base_rank_effective_dim_floor=float(base_rank_effective_dim_floor),
            base_svd_rank_floor=float(base_svd_rank_floor),
            factor_geometry_available=bool(factor_rows.numel() and factor_rows.size(0) >= 2),
        )
        if selection_mode == "train_proxy" and (selected_score is None or final_epoch_score > selected_score):
            selected_score = float(final_epoch_score)
            selected_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if selection_mode == "train_proxy" and best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        final_outputs = model(input_pack)
        final_losses = matryoshka_signature_losses(final_outputs, train_pack)
        final_aux_losses = _metamer_case_losses(
            final_outputs.h,
            packet_case_ids=packet_case_ids,
            separation_margin=float(separation_margin),
        )
        final_run_losses = _run_center_contrastive_losses(
            final_outputs.h,
            packet_case_ids=packet_case_ids,
            packet_run_ids=packet_run_ids,
            temperature=float(run_contrastive_temperature),
        )
        final_rank_losses = _rank_entropy_losses(final_outputs.h, entropy_floor=float(rank_entropy_floor))
        final_run_rank_losses = _run_center_rank_entropy_losses(
            final_outputs.h,
            packet_run_ids=packet_run_ids,
            effective_dim_floor=float(run_rank_effective_dim_floor),
        )
        final_base_rank_losses = _effective_dim_floor_losses(
            final_outputs.h[packet_is_base],
            effective_dim_floor=float(base_rank_effective_dim_floor),
            prefix="base_rank",
        )
        final_base_svd_losses = _svd_effective_rank_losses(
            final_outputs.h[packet_is_base],
            effective_rank_floor=float(base_svd_rank_floor),
            prefix="base_svd_rank",
        )
        final_factor_geometry_losses = _factor_geometry_losses(final_outputs.h, factor_rows)
        final_weighted = _weighted_training_objective(
            final_losses,
            final_aux_losses,
            final_run_losses,
            final_rank_losses,
            final_run_rank_losses,
            final_base_rank_losses,
            final_base_svd_losses,
            metamer_consistency_weight=metamer_consistency_weight,
            case_separation_weight=case_separation_weight,
            run_contrastive_weight=run_contrastive_weight,
            rank_entropy_weight=rank_entropy_weight,
            run_rank_entropy_weight=run_rank_entropy_weight,
            base_rank_entropy_weight=base_rank_entropy_weight,
            base_svd_rank_weight=base_svd_rank_weight,
        )
        final_weighted = final_weighted + float(factor_geometry_weight) * final_factor_geometry_losses[
            "factor_geometry_loss"
        ]
        contract = summarize_learning_contract(final_outputs, final_losses)
    return model, {
        "train_packet_count": count,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "selection_objective": selection_mode,
        "selected_epoch": int(selected_epoch),
        "selected_score": float(selected_score if selected_score is not None else final_epoch_score),
        "input_redaction": "law_signature_and_operator_seed_zeroed_for_encoder_input",
        "target_redaction": "none",
        "initial_batch_loss": float(initial_loss if initial_loss is not None else 0.0),
        "final_full_loss": _scalar(final_weighted),
        "loss_reduction": float((initial_loss if initial_loss is not None else 0.0) - _scalar(final_weighted)),
        "final_matryoshka_loss": _scalar(final_losses["total_loss"]),
        "final_law_signature_loss": _scalar(final_losses["law_signature_loss"]),
        "final_operator_seed_loss": _scalar(final_losses["operator_seed_loss"]),
        "final_tail_law_signature_loss": _scalar(final_losses["tail_law_signature_loss"]),
        "final_tail_operator_seed_loss": _scalar(final_losses["tail_operator_seed_loss"]),
        "final_tail_operator_alignment_loss": _scalar(final_losses["tail_operator_alignment_loss"]),
        "final_run_positive_cosine": _scalar(final_run_losses["run_positive_cosine"]),
        "final_run_negative_cosine": _scalar(final_run_losses["run_negative_cosine"]),
        "final_rank_entropy_effective_dim": _scalar(final_rank_losses["rank_entropy_effective_dim"]),
        "final_run_rank_effective_dim": _scalar(final_run_rank_losses["run_rank_effective_dim"]),
        "final_base_rank_effective_dim": _scalar(final_base_rank_losses["base_rank_effective_dim"]),
        "final_base_svd_rank_effective_dim": _scalar(final_base_svd_losses["base_svd_rank_effective_dim"]),
        "final_factor_geometry_cosine": _scalar(final_factor_geometry_losses["factor_geometry_cosine"]),
        "losses_by_epoch": losses_by_epoch,
        "contract": contract,
    }


def _feature_views(
    *,
    metadata: torch.Tensor,
    h: torch.Tensor,
    case_ids: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    tail = h[:, 640:768].detach().float()
    prefix = h[:, :640].detach().float()
    full_h = h.detach().float()
    return {
        "metadata_only": metadata.detach().float(),
        "metadata_plus_tail": _cat_feature(metadata, tail),
        "tail_only": tail,
        "metadata_plus_full_h": _cat_feature(metadata, full_h),
        "metadata_plus_prefix": _cat_feature(metadata, prefix),
        "metadata_plus_permuted_tail": _cat_feature(metadata, _permuted_rows(tail, seed=seed + 101)),
        "metadata_plus_cross_case_tail": _cat_feature(metadata, _cross_case_rows(tail, case_ids)),
        "metadata_plus_random_tail": _cat_feature(metadata, _random_rows_like(tail, seed=seed + 202)),
    }


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
                "heldout_r2_vs_train_mean": _relative_improvement(baseline, heldout_mse),
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


def _best(matrix: dict[str, Any], target_name: str, view_name: str) -> dict[str, Any]:
    return matrix[target_name]["views"][view_name]["best_by_heldout_mse"]


def _summarize_incremental(matrix: dict[str, Any], targets: tuple[str, ...]) -> dict[str, Any]:
    target_rows = []
    tail_win_count = 0
    tail_beats_prefix_count = 0
    tail_beats_permuted_count = 0
    tail_beats_cross_case_count = 0
    tail_beats_random_count = 0
    shuffle_control_win_count = 0
    prefix_control_win_count = 0
    cross_case_control_win_count = 0
    random_control_win_count = 0
    for target_name in targets:
        meta = _best(matrix, target_name, "metadata_only")
        tail = _best(matrix, target_name, "metadata_plus_tail")
        prefix = _best(matrix, target_name, "metadata_plus_prefix")
        permuted = _best(matrix, target_name, "metadata_plus_permuted_tail")
        cross_case = _best(matrix, target_name, "metadata_plus_cross_case_tail")
        random_tail = _best(matrix, target_name, "metadata_plus_random_tail")
        full_h = _best(matrix, target_name, "metadata_plus_full_h")
        tail_mse = _finite_float(tail.get("heldout_mse"), float("inf"))
        meta_mse = _finite_float(meta.get("heldout_mse"), float("inf"))
        prefix_mse = _finite_float(prefix.get("heldout_mse"), float("inf"))
        permuted_mse = _finite_float(permuted.get("heldout_mse"), float("inf"))
        cross_mse = _finite_float(cross_case.get("heldout_mse"), float("inf"))
        random_mse = _finite_float(random_tail.get("heldout_mse"), float("inf"))
        full_h_mse = _finite_float(full_h.get("heldout_mse"), float("inf"))
        tail_r2 = _finite_float(tail.get("heldout_r2_vs_train_mean"))
        meta_r2 = _finite_float(meta.get("heldout_r2_vs_train_mean"))
        tail_win = tail_mse + EPS < meta_mse
        tail_beats_prefix = tail_mse + EPS < prefix_mse
        tail_beats_permuted = tail_mse + EPS < permuted_mse
        tail_beats_cross = tail_mse + EPS < cross_mse
        tail_beats_random = tail_mse + EPS < random_mse
        shuffle_control_win = permuted_mse + EPS < meta_mse
        prefix_control_win = prefix_mse + EPS < meta_mse
        cross_case_control_win = cross_mse + EPS < meta_mse
        random_control_win = random_mse + EPS < meta_mse
        tail_win_count += int(tail_win)
        tail_beats_prefix_count += int(tail_beats_prefix)
        tail_beats_permuted_count += int(tail_beats_permuted)
        tail_beats_cross_case_count += int(tail_beats_cross)
        tail_beats_random_count += int(tail_beats_random)
        shuffle_control_win_count += int(shuffle_control_win)
        prefix_control_win_count += int(prefix_control_win)
        cross_case_control_win_count += int(cross_case_control_win)
        random_control_win_count += int(random_control_win)
        target_rows.append(
            {
                "target": target_name,
                "metadata_heldout_mse": meta_mse,
                "tail_heldout_mse": tail_mse,
                "prefix_heldout_mse": prefix_mse,
                "permuted_tail_heldout_mse": permuted_mse,
                "cross_case_tail_heldout_mse": cross_mse,
                "random_tail_heldout_mse": random_mse,
                "full_h_heldout_mse": full_h_mse,
                "delta_heldout_mse_vs_metadata": meta_mse - tail_mse,
                "delta_heldout_r2_vs_metadata": tail_r2 - meta_r2,
                "tail_minus_prefix_delta_mse": prefix_mse - tail_mse,
                "tail_minus_permuted_delta_mse": permuted_mse - tail_mse,
                "tail_minus_cross_case_delta_mse": cross_mse - tail_mse,
                "tail_minus_random_delta_mse": random_mse - tail_mse,
                "tail_win": tail_win,
                "tail_beats_prefix": tail_beats_prefix,
                "tail_beats_permuted": tail_beats_permuted,
                "tail_beats_cross_case": tail_beats_cross,
                "tail_beats_random": tail_beats_random,
                "shuffle_control_win": shuffle_control_win,
                "prefix_control_win": prefix_control_win,
                "cross_case_control_win": cross_case_control_win,
                "random_control_win": random_control_win,
            }
        )
    target_count = len(targets)
    control_win_count = (
        shuffle_control_win_count
        + prefix_control_win_count
        + cross_case_control_win_count
        + random_control_win_count
    )
    if (
        target_count
        and tail_win_count == target_count
        and tail_beats_prefix_count == target_count
        and tail_beats_permuted_count == target_count
        and tail_beats_cross_case_count == target_count
        and tail_beats_random_count == target_count
        and shuffle_control_win_count == 0
        and prefix_control_win_count == 0
        and cross_case_control_win_count == 0
        and random_control_win_count == 0
    ):
        status = "tail_incremental_usefulness_strong"
    elif (
        target_count
        and tail_win_count >= max(1, math.ceil(target_count / 2))
        and tail_beats_permuted_count >= max(1, math.ceil(target_count / 2))
        and tail_beats_cross_case_count >= max(1, math.ceil(target_count / 2))
        and tail_beats_random_count >= max(1, math.ceil(target_count / 2))
    ):
        status = (
            "tail_incremental_usefulness_mixed_controlled"
            if control_win_count == 0
            else "tail_incremental_usefulness_mixed_control_confounded"
        )
    else:
        status = "tail_incremental_not_supported"
    return {
        "status": status,
        "target_count": target_count,
        "tail_win_count": tail_win_count,
        "tail_beats_prefix_count": tail_beats_prefix_count,
        "tail_beats_permuted_count": tail_beats_permuted_count,
        "tail_beats_cross_case_count": tail_beats_cross_case_count,
        "tail_beats_random_count": tail_beats_random_count,
        "shuffle_control_win_count": shuffle_control_win_count,
        "prefix_control_win_count": prefix_control_win_count,
        "cross_case_control_win_count": cross_case_control_win_count,
        "random_control_win_count": random_control_win_count,
        "control_win_count": control_win_count,
        "target_rows": target_rows,
    }


def evaluate_signature_tail_incremental_usefulness(
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
        raise RuntimeError("tail incremental usefulness requires non-empty train and heldout case sets")
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
    train_packs, train_case_ids, train_run_ids, train_is_base, _train_labels = _collect_packs(
        train_case_runs,
        device=device,
    )
    heldout_packs, heldout_case_ids, _heldout_run_ids, _heldout_is_base, _heldout_labels = _collect_packs(
        heldout_case_runs,
        device=device,
    )
    train_pack = _concat_factor_packs(train_packs)
    heldout_pack = _concat_factor_packs(heldout_packs)
    train_case_tensor = torch.tensor(train_case_ids, dtype=torch.long, device=device)
    heldout_case_tensor = torch.tensor(heldout_case_ids, dtype=torch.long, device=device)
    _set_rng_seed(int(seed))
    model, train_summary = _train_non_law_input_model(
        train_pack,
        packet_case_ids=train_case_tensor,
        packet_run_ids=torch.tensor(train_run_ids, dtype=torch.long, device=device),
        packet_is_base=torch.tensor(train_is_base, dtype=torch.bool, device=device),
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
    train_h = _learned_rows_from_non_law_input(model, train_pack, device)
    heldout_h = _learned_rows_from_non_law_input(model, heldout_pack, device)
    train_metadata = _metadata_features(train_pack)
    heldout_metadata = _metadata_features(heldout_pack)
    train_views = _feature_views(
        metadata=train_metadata,
        h=train_h,
        case_ids=train_case_tensor,
        seed=int(seed),
    )
    heldout_views = _feature_views(
        metadata=heldout_metadata,
        h=heldout_h,
        case_ids=heldout_case_tensor,
        seed=int(seed) + 1000,
    )
    train_targets = _target_views(train_pack)
    heldout_targets = _target_views(heldout_pack)
    matrix = _probe_matrix(
        train_views=train_views,
        heldout_views=heldout_views,
        train_targets=train_targets,
        heldout_targets=heldout_targets,
        targets=targets,
        ridge_alphas=ridge_alphas,
    )
    incremental = _summarize_incremental(matrix, targets)
    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "runtime": "circleworld_proto",
        "purpose": "bounded_operator_tail_incremental_usefulness_probe",
        "status": incremental["status"],
        "promotion_effect": "none",
        "interpretation": (
            "This is a conditional usefulness probe, not an audio proof. It asks whether learned operator-tail "
            "features add heldout law/operator predictive information beyond explicit non-law metadata. The encoder "
            "input redacts law_signature and operator_seed, so the tail cannot directly copy the targets. Strong "
            "wins are rejected when they also appear in shuffled, cross-case, random, or prefix controls."
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
        "train_packet_count": int(train_pack.q_profile.size(0)),
        "heldout_packet_count": int(heldout_pack.q_profile.size(0)),
        "targets": list(targets),
        "ridge_alphas": [float(alpha) for alpha in ridge_alphas],
        "ridge_alpha_selection_policy": "fixed" if len(ridge_alphas) == 1 else "heldout_mse_best_diagnostic",
        "leakage_guard": {
            "encoder_law_signature_input": "zeroed",
            "encoder_operator_seed_input": "zeroed",
            "target_law_signature": "real",
            "target_operator_seed": "real",
        },
        "feature_views": {
            "metadata_only": "explicit coarse/q/arc/temporal/support/branch/confidence factors only",
            "metadata_plus_tail": "metadata plus learned h[:,640:768] from redacted non-law encoder input",
            "tail_only": "learned h[:,640:768] only from redacted non-law encoder input",
            "metadata_plus_full_h": "metadata plus all learned h from redacted non-law encoder input",
            "metadata_plus_prefix": "metadata plus learned h[:640] from redacted non-law encoder input",
            "metadata_plus_permuted_tail": "metadata plus row-permuted tail",
            "metadata_plus_cross_case_tail": "metadata plus tail copied from another case where possible",
            "metadata_plus_random_tail": "metadata plus random same-shape tail",
        },
        "train_summary": train_summary,
        "probe_matrix": matrix,
        "incremental_summary": incremental,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "signature_tail_incremental_usefulness.json"
    md_path = out_dir / "SIGNATURE_TAIL_INCREMENTAL_USEFULNESS.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    inc = summary["incremental_summary"]
    train = summary["train_summary"]
    lines = [
        "# Signature Tail Incremental Usefulness",
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
        f"- train packets: `{summary['train_packet_count']}`",
        f"- heldout packets: `{summary['heldout_packet_count']}`",
        f"- input redaction: `{train.get('input_redaction')}`",
        f"- ridge alpha policy: `{summary.get('ridge_alpha_selection_policy')}`",
        f"- final law signature loss: `{train.get('final_law_signature_loss')}`",
        f"- final operator seed loss: `{train.get('final_operator_seed_loss')}`",
        f"- final tail law signature loss: `{train.get('final_tail_law_signature_loss')}`",
        f"- final tail operator seed loss: `{train.get('final_tail_operator_seed_loss')}`",
        f"- final tail operator alignment loss: `{train.get('final_tail_operator_alignment_loss')}`",
        f"- tail wins: `{inc['tail_win_count']}` / `{inc['target_count']}`",
        f"- tail beats prefix: `{inc['tail_beats_prefix_count']}` / `{inc['target_count']}`",
        f"- tail beats permuted: `{inc['tail_beats_permuted_count']}` / `{inc['target_count']}`",
        f"- tail beats cross-case: `{inc['tail_beats_cross_case_count']}` / `{inc['target_count']}`",
        f"- tail beats random: `{inc['tail_beats_random_count']}` / `{inc['target_count']}`",
        f"- shuffle control wins: `{inc['shuffle_control_win_count']}` / `{inc['target_count']}`",
        f"- prefix control wins: `{inc.get('prefix_control_win_count', 0)}` / `{inc['target_count']}`",
        f"- cross-case control wins: `{inc.get('cross_case_control_win_count', 0)}` / `{inc['target_count']}`",
        f"- random control wins: `{inc.get('random_control_win_count', 0)}` / `{inc['target_count']}`",
        f"- total control wins: `{inc.get('control_win_count', 0)}`",
        "",
        "## Incremental Targets",
        "",
        "| target | tail win | d mse vs metadata | d r2 vs metadata | tail-prefix d mse | tail-shuffle d mse | tail-cross d mse | tail-random d mse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in inc.get("target_rows", []):
        lines.append(
            "| {target} | {win} | {dmse:.6g} | {dr2:.6g} | {dprefix:.6g} | {dperm:.6g} | {dcross:.6g} | {drand:.6g} |".format(
                target=row.get("target"),
                win="yes" if row.get("tail_win") else "no",
                dmse=_finite_float(row.get("delta_heldout_mse_vs_metadata")),
                dr2=_finite_float(row.get("delta_heldout_r2_vs_metadata")),
                dprefix=_finite_float(row.get("tail_minus_prefix_delta_mse")),
                dperm=_finite_float(row.get("tail_minus_permuted_delta_mse")),
                dcross=_finite_float(row.get("tail_minus_cross_case_delta_mse")),
                drand=_finite_float(row.get("tail_minus_random_delta_mse")),
            )
        )
    lines.extend(["", "## Best Probe Rows", ""])
    for target_name, target_row in summary["probe_matrix"].items():
        lines.extend([f"### {target_name}", ""])
        lines.append("| view | alpha | heldout mse | heldout r2 |")
        lines.append("|---|---:|---:|---:|")
        rows = []
        for view_name, view_row in target_row["views"].items():
            best = view_row["best_by_heldout_mse"]
            rows.append((view_name, best))
        rows.sort(key=lambda item: _finite_float(item[1].get("heldout_mse"), float("inf")))
        for view_name, best in rows:
            lines.append(
                "| {view} | {alpha} | {mse:.6g} | {r2:.6g} |".format(
                    view=view_name,
                    alpha=best.get("ridge_alpha"),
                    mse=_finite_float(best.get("heldout_mse")),
                    r2=_finite_float(best.get("heldout_r2_vs_train_mean")),
                )
            )
        lines.append("")
    lines.extend(["## Interpretation", "", summary["interpretation"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether learned operator-tail features add heldout law/operator information beyond metadata."
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
    summary = evaluate_signature_tail_incremental_usefulness(
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
                "saved": str(Path(args.out_dir) / "signature_tail_incremental_usefulness.json"),
                "report": str(Path(args.out_dir) / "SIGNATURE_TAIL_INCREMENTAL_USEFULNESS.md"),
                "status": summary["status"],
                "tail_win_count": summary["incremental_summary"]["tail_win_count"],
                "tail_beats_prefix_count": summary["incremental_summary"]["tail_beats_prefix_count"],
                "tail_beats_permuted_count": summary["incremental_summary"]["tail_beats_permuted_count"],
                "target_count": summary["incremental_summary"]["target_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
