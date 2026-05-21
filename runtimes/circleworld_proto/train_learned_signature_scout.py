from __future__ import annotations

import argparse
import json
import random
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
from evaluate_dense_signature_claim import (  # noqa: E402
    DEFAULT_CLUSTER_THRESHOLD,
    _concat_rows,
    _cos,
    _dense_prefix_metrics,
    _dense_rows,
    _empty_rows,
    _mean,
    _metadata_rows,
    _pairwise_summary,
    _pooled_rows,
    _representation_summary,
    _select_cases,
    _stddev,
)
from evaluate_relational_metamers import (  # noqa: E402
    _apply_transform,
    _default_cases_path,
    _load_cases,
    _run_circleworld,
    _safe_device,
    _transform_specs,
)
from export_circleworld_audio import _load_circle_cfg, prepare_reference_audio  # noqa: E402
from rafa_math_tools import phase_to_phasor  # noqa: E402
from rafa_relational_signature import (  # noqa: E402
    RelationalFactorPack,
    extract_relational_factor_pack,
)
from rafa_relational_signature_learning import (  # noqa: E402
    RafaRelationalSignatureEncoderV0,
    factor_pack_to_learning_input,
    matryoshka_signature_losses,
    summarize_learning_contract,
)
from stft_utils import compute_stft  # noqa: E402


SCHEMA = "rafa_learned_signature_scout_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "learned_signature_scout_2026_05_07"
DEFAULT_CONFIG = (
    TOKENBURST_ROOT
    / "child_writeback_ablation_v4_seed_fallback_cuda_2026_05_06"
    / "support_operator_floor_high"
    / "circleworld_config.json"
)
DEFAULT_BROAD18_CASES = TOKENBURST_ROOT / "broad_wav_cases_18_2026_05_06.json"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            return float(value.detach().cpu().mean().item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return float(default)


def _set_rng_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _weighted_training_objective(
    losses: dict[str, torch.Tensor],
    aux_losses: dict[str, torch.Tensor],
    run_losses: dict[str, torch.Tensor],
    rank_losses: dict[str, torch.Tensor],
    run_rank_losses: dict[str, torch.Tensor],
    base_rank_losses: dict[str, torch.Tensor],
    base_svd_losses: dict[str, torch.Tensor],
    *,
    metamer_consistency_weight: float,
    case_separation_weight: float,
    run_contrastive_weight: float,
    rank_entropy_weight: float,
    run_rank_entropy_weight: float,
    base_rank_entropy_weight: float,
    base_svd_rank_weight: float,
) -> torch.Tensor:
    return (
        losses["total_loss"]
        + float(metamer_consistency_weight) * aux_losses["metamer_consistency_loss"]
        + float(case_separation_weight) * aux_losses["case_separation_loss"]
        + float(run_contrastive_weight) * run_losses["run_contrastive_loss"]
        + float(rank_entropy_weight) * rank_losses["rank_entropy_loss"]
        + float(run_rank_entropy_weight) * run_rank_losses["run_rank_entropy_loss"]
        + float(base_rank_entropy_weight) * base_rank_losses["base_rank_entropy_loss"]
        + float(base_svd_rank_weight) * base_svd_losses["base_svd_rank_loss"]
    )


def _signature_loss_bounds(spec: str, width: int) -> tuple[int, int]:
    name = str(spec or "full").strip().lower()
    dim = max(1, int(width))
    if name == "full":
        return 0, dim
    if name in {"prefix640", "prefix:640"}:
        return 0, max(1, min(640, dim))
    if name in {"tail640", "tail:640"}:
        start = max(0, min(640, dim - 1))
        return start, dim
    if name.startswith("prefix:"):
        end = int(name.split(":", 1)[1])
        return 0, max(1, min(end, dim))
    if name.startswith("tail:"):
        start = int(name.split(":", 1)[1])
        return max(0, min(start, dim - 1)), dim
    if name.startswith("span:"):
        start_s, end_s = name.split(":", 1)[1].split(":", 1)
        start = max(0, min(int(start_s), dim - 1))
        end = max(start + 1, min(int(end_s), dim))
        return start, end
    raise ValueError(f"unsupported signature loss view: {spec!r}")


def _signature_loss_view(h: torch.Tensor, view: str) -> torch.Tensor:
    start, end = _signature_loss_bounds(str(view or "full"), int(h.size(-1)))
    return h[..., start:end]


def _bounded_floor_ratio(value: torch.Tensor, floor: float) -> float:
    if float(floor) <= 0.0:
        return 1.0
    return max(0.0, min(_as_float(value), float(floor)) / max(float(floor), 1.0e-8))


def _train_proxy_selection_score(
    aux_losses: dict[str, torch.Tensor],
    run_losses: dict[str, torch.Tensor],
    base_rank_losses: dict[str, torch.Tensor],
    base_svd_losses: dict[str, torch.Tensor],
    factor_geometry_losses: dict[str, torch.Tensor],
    *,
    base_rank_effective_dim_floor: float,
    base_svd_rank_floor: float,
    factor_geometry_available: bool,
) -> float:
    components = [
        -_as_float(aux_losses["metamer_consistency_loss"]),
        (
            _as_float(run_losses["run_positive_cosine"])
            - _as_float(run_losses["run_negative_cosine"])
            - _as_float(run_losses["run_contrastive_loss"])
        ),
        (
            _bounded_floor_ratio(base_rank_losses["base_rank_effective_dim"], base_rank_effective_dim_floor)
            - _as_float(base_rank_losses["base_rank_entropy_loss"])
        ),
        (
            _bounded_floor_ratio(base_svd_losses["base_svd_rank_effective_dim"], base_svd_rank_floor)
            - _as_float(base_svd_losses["base_svd_rank_loss"])
        ),
    ]
    if factor_geometry_available:
        components.append(
            _as_float(factor_geometry_losses["factor_geometry_cosine"])
            - _as_float(factor_geometry_losses["factor_geometry_loss"])
        )
    return float(sum(components) / max(1, len(components)))


def _default_cases_for_scout() -> Path:
    return DEFAULT_BROAD18_CASES if DEFAULT_BROAD18_CASES.exists() else _default_cases_path()


def _select_case_slice(cases: dict[str, str], *, start: int, count: int | None) -> dict[str, str]:
    items = list(cases.items())
    start_idx = max(0, int(start))
    if count is None or int(count) <= 0:
        return dict(items[start_idx:])
    return dict(items[start_idx : start_idx + int(count)])


def _rotate_cases(cases: dict[str, str], offset: int) -> dict[str, str]:
    items = list(cases.items())
    if not items:
        return {}
    shift = int(offset) % len(items)
    return dict([*items[shift:], *items[:shift]])


def _shuffle_cases(cases: dict[str, str], shuffle_seed: int | None) -> dict[str, str]:
    items = list(cases.items())
    if shuffle_seed is None:
        return dict(items)
    rng = random.Random(int(shuffle_seed))
    rng.shuffle(items)
    return dict(items)


def _concat_factor_packs(packs: list[RelationalFactorPack]) -> RelationalFactorPack:
    kept = [pack for pack in packs if pack.q_profile.numel() > 0]
    if not kept:
        raise ValueError("no non-empty relational factor packs to concatenate")
    return RelationalFactorPack(
        coarse_profile=torch.cat([pack.coarse_profile.detach().float() for pack in kept], dim=0),
        q_profile=torch.cat([pack.q_profile.detach().float() for pack in kept], dim=0),
        arc_profile=torch.cat([pack.arc_profile.detach().float() for pack in kept], dim=0),
        temporal_profile=torch.cat([pack.temporal_profile.detach().float() for pack in kept], dim=0),
        support_profile=torch.cat([pack.support_profile.detach().float() for pack in kept], dim=0),
        branch_profile=torch.cat([pack.branch_profile.detach().float() for pack in kept], dim=0),
        law_signature=torch.cat([pack.law_signature.detach().float() for pack in kept], dim=0),
        confidence=torch.cat([pack.confidence.detach().float() for pack in kept], dim=0),
        operator_seed=torch.cat([pack.operator_seed.detach().float() for pack in kept], dim=0),
    ).validate()


def _factor_pack_for_run(run: dict[str, Any], device: torch.device) -> RelationalFactorPack | None:
    pack, *_rest = extract_relational_factor_pack(run)
    if pack is None:
        return None
    return RelationalFactorPack(
        coarse_profile=pack.coarse_profile.detach().float().to(device),
        q_profile=pack.q_profile.detach().float().to(device),
        arc_profile=pack.arc_profile.detach().float().to(device),
        temporal_profile=pack.temporal_profile.detach().float().to(device),
        support_profile=pack.support_profile.detach().float().to(device),
        branch_profile=pack.branch_profile.detach().float().to(device),
        law_signature=pack.law_signature.detach().float().to(device),
        confidence=pack.confidence.detach().float().to(device),
        operator_seed=pack.operator_seed.detach().float().to(device),
    ).validate()


def _learned_rows(
    model: RafaRelationalSignatureEncoderV0,
    pack: RelationalFactorPack | None,
    device: torch.device,
) -> torch.Tensor:
    if pack is None or pack.q_profile.numel() == 0:
        return _empty_rows(device=device)
    with torch.no_grad():
        outputs = model(pack)
    return outputs.h.detach().float()


def _prefix_metrics_from_rows(
    base_rows: torch.Tensor,
    transformed_rows: torch.Tensor,
    prefix_dims: tuple[int, ...],
) -> dict[str, Any]:
    return _dense_prefix_metrics(base_rows, transformed_rows, prefix_dims)


def _case_runs(
    *,
    case_name: str,
    wav_path: Path,
    cfg: Any,
    stft_cfg: dict[str, Any],
    device: torch.device,
    device_name: str,
    clip_seconds: int,
    transforms: list[dict[str, Any]],
) -> dict[str, Any]:
    wav, _sr = prepare_reference_audio(
        wav_path=wav_path,
        device_name=device_name,
        clip_seconds_override=clip_seconds,
    )
    wav = wav.to(device)
    _mag, phase = compute_stft(wav, stft_cfg)
    phase_state = phase_to_phasor(phase)
    base_run = _run_circleworld(phase_state, cfg)
    transform_runs = []
    for transform in transforms:
        phase_state_t = _apply_transform(
            wav=wav,
            phase_state=phase_state,
            transform=transform,
            stft_cfg=stft_cfg,
        )
        transform_runs.append({"spec": transform, "run": _run_circleworld(phase_state_t, cfg)})
    return {
        "case": case_name,
        "wav_path": str(wav_path),
        "base_run": base_run,
        "transform_runs": transform_runs,
    }


def _train_model(
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
    metamer_loss_view: str = "full",
    case_separation_loss_view: str = "full",
    run_contrastive_loss_view: str = "full",
    redact_law_operator_inputs: bool = False,
    selection_objective: str = "final",
) -> tuple[RafaRelationalSignatureEncoderV0, dict[str, Any]]:
    device = train_pack.q_profile.device
    selection_mode = str(selection_objective)
    if selection_mode not in {"final", "train_proxy"}:
        raise ValueError(f"unsupported selection objective: {selection_objective!r}")
    for view_name in (metamer_loss_view, case_separation_loss_view, run_contrastive_loss_view):
        _signature_loss_bounds(str(view_name or "full"), 768)
    model = RafaRelationalSignatureEncoderV0.from_factor_pack(
        train_pack,
        hidden_dim=int(hidden_dim),
        redact_law_operator_input=bool(redact_law_operator_inputs),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    _x = factor_pack_to_learning_input(train_pack, model.cfg)
    count = int(_x.size(0))
    epoch_count = max(1, int(epochs))
    factor_geometry_available = bool(_x.numel() > 0 and _x.size(0) == count and count >= 2)
    packet_case_ids = packet_case_ids.to(device=device, dtype=torch.long)
    packet_run_ids = packet_run_ids.to(device=device, dtype=torch.long)
    packet_is_base = packet_is_base.to(device=device, dtype=torch.bool)
    losses_by_epoch: list[float] = []
    aux_by_epoch: list[dict[str, float]] = []
    selected_epoch = epoch_count
    selected_score: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    initial_loss = None
    for epoch in range(epoch_count):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(train_pack)
        losses = matryoshka_signature_losses(outputs, train_pack)
        aux_losses = _metamer_case_losses_for_views(
            outputs.h,
            packet_case_ids=packet_case_ids,
            separation_margin=float(separation_margin),
            metamer_loss_view=str(metamer_loss_view),
            case_separation_loss_view=str(case_separation_loss_view),
        )
        run_losses = _run_center_contrastive_losses(
            _signature_loss_view(outputs.h, run_contrastive_loss_view),
            packet_case_ids=packet_case_ids,
            packet_run_ids=packet_run_ids,
            temperature=float(run_contrastive_temperature),
        )
        rank_losses = _rank_entropy_losses(
            outputs.h,
            entropy_floor=float(rank_entropy_floor),
        )
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
        factor_geometry_losses = _factor_geometry_losses(outputs.h, _x)
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
        loss_value = float(loss.detach().cpu().item())
        if initial_loss is None:
            initial_loss = loss_value
        losses_by_epoch.append(loss_value)
        epoch_aux = {key: float(value.detach().cpu().item()) for key, value in aux_losses.items()}
        epoch_aux.update({key: float(value.detach().cpu().item()) for key, value in run_losses.items()})
        epoch_aux.update({key: float(value.detach().cpu().item()) for key, value in rank_losses.items()})
        epoch_aux.update({key: float(value.detach().cpu().item()) for key, value in run_rank_losses.items()})
        epoch_aux.update({key: float(value.detach().cpu().item()) for key, value in base_rank_losses.items()})
        epoch_aux.update({key: float(value.detach().cpu().item()) for key, value in base_svd_losses.items()})
        epoch_aux.update({key: float(value.detach().cpu().item()) for key, value in factor_geometry_losses.items()})
        aux_by_epoch.append(epoch_aux)
        if selection_mode == "train_proxy":
            with torch.no_grad():
                selection_outputs = model(train_pack)
                selection_aux_losses = _metamer_case_losses_for_views(
                    selection_outputs.h,
                    packet_case_ids=packet_case_ids,
                    separation_margin=float(separation_margin),
                    metamer_loss_view=str(metamer_loss_view),
                    case_separation_loss_view=str(case_separation_loss_view),
                )
                selection_run_losses = _run_center_contrastive_losses(
                    _signature_loss_view(selection_outputs.h, run_contrastive_loss_view),
                    packet_case_ids=packet_case_ids,
                    packet_run_ids=packet_run_ids,
                    temperature=float(run_contrastive_temperature),
                )
                selection_base_rank_losses = _effective_dim_floor_losses(
                    selection_outputs.h[packet_is_base],
                    effective_dim_floor=float(base_rank_effective_dim_floor),
                    prefix="base_rank",
                )
                selection_base_svd_losses = _svd_effective_rank_losses(
                    selection_outputs.h[packet_is_base],
                    effective_rank_floor=float(base_svd_rank_floor),
                    prefix="base_svd_rank",
                )
                selection_factor_geometry_losses = _factor_geometry_losses(selection_outputs.h, _x)
                epoch_score = _train_proxy_selection_score(
                    selection_aux_losses,
                    selection_run_losses,
                    selection_base_rank_losses,
                    selection_base_svd_losses,
                    selection_factor_geometry_losses,
                    base_rank_effective_dim_floor=float(base_rank_effective_dim_floor),
                    base_svd_rank_floor=float(base_svd_rank_floor),
                    factor_geometry_available=factor_geometry_available,
                )
            if selected_score is None or epoch_score > selected_score:
                selected_epoch = epoch + 1
                selected_score = epoch_score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        final_outputs = model(train_pack)
        final_losses = matryoshka_signature_losses(final_outputs, train_pack)
        final_aux_losses = _metamer_case_losses_for_views(
            final_outputs.h,
            packet_case_ids=packet_case_ids,
            separation_margin=float(separation_margin),
            metamer_loss_view=str(metamer_loss_view),
            case_separation_loss_view=str(case_separation_loss_view),
        )
        final_run_losses = _run_center_contrastive_losses(
            _signature_loss_view(final_outputs.h, run_contrastive_loss_view),
            packet_case_ids=packet_case_ids,
            packet_run_ids=packet_run_ids,
            temperature=float(run_contrastive_temperature),
        )
        final_rank_losses = _rank_entropy_losses(
            final_outputs.h,
            entropy_floor=float(rank_entropy_floor),
        )
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
        final_factor_geometry_losses = _factor_geometry_losses(final_outputs.h, _x)
        final_epoch_score = _train_proxy_selection_score(
            final_aux_losses,
            final_run_losses,
            final_base_rank_losses,
            final_base_svd_losses,
            final_factor_geometry_losses,
            base_rank_effective_dim_floor=float(base_rank_effective_dim_floor),
            base_svd_rank_floor=float(base_svd_rank_floor),
            factor_geometry_available=factor_geometry_available,
        )
    if selection_mode == "train_proxy" and best_state is not None:
        selected_score = float(selected_score if selected_score is not None else final_epoch_score)
        model.load_state_dict(best_state)
        with torch.no_grad():
            final_outputs = model(train_pack)
            final_losses = matryoshka_signature_losses(final_outputs, train_pack)
            final_aux_losses = _metamer_case_losses_for_views(
                final_outputs.h,
                packet_case_ids=packet_case_ids,
                separation_margin=float(separation_margin),
                metamer_loss_view=str(metamer_loss_view),
                case_separation_loss_view=str(case_separation_loss_view),
            )
            final_run_losses = _run_center_contrastive_losses(
                _signature_loss_view(final_outputs.h, run_contrastive_loss_view),
                packet_case_ids=packet_case_ids,
                packet_run_ids=packet_run_ids,
                temperature=float(run_contrastive_temperature),
            )
            final_rank_losses = _rank_entropy_losses(
                final_outputs.h,
                entropy_floor=float(rank_entropy_floor),
            )
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
            final_factor_geometry_losses = _factor_geometry_losses(final_outputs.h, _x)
    else:
        selected_score = float(final_epoch_score)
    contract = summarize_learning_contract(final_outputs, final_losses)
    final_matryoshka_loss = float(final_losses["total_loss"].detach().cpu().item())
    final_weighted_loss = _weighted_training_objective(
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
    final_weighted_loss = final_weighted_loss + float(factor_geometry_weight) * final_factor_geometry_losses[
        "factor_geometry_loss"
    ]
    final_loss = float(final_weighted_loss.detach().cpu().item())
    return model, {
        "train_packet_count": count,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "metamer_consistency_weight": float(metamer_consistency_weight),
        "case_separation_weight": float(case_separation_weight),
        "separation_margin": float(separation_margin),
        "run_contrastive_weight": float(run_contrastive_weight),
        "run_contrastive_temperature": float(run_contrastive_temperature),
        "rank_entropy_weight": float(rank_entropy_weight),
        "rank_entropy_floor": float(rank_entropy_floor),
        "run_rank_entropy_weight": float(run_rank_entropy_weight),
        "run_rank_effective_dim_floor": float(run_rank_effective_dim_floor),
        "base_rank_entropy_weight": float(base_rank_entropy_weight),
        "base_rank_effective_dim_floor": float(base_rank_effective_dim_floor),
        "base_svd_rank_weight": float(base_svd_rank_weight),
        "base_svd_rank_floor": float(base_svd_rank_floor),
        "factor_geometry_weight": float(factor_geometry_weight),
        "metamer_loss_view": str(metamer_loss_view),
        "case_separation_loss_view": str(case_separation_loss_view),
        "run_contrastive_loss_view": str(run_contrastive_loss_view),
        "metamer_consistency_slice": str(metamer_loss_view),
        "case_separation_slice": str(case_separation_loss_view),
        "run_contrastive_slice": str(run_contrastive_loss_view),
        "redact_law_operator_inputs": bool(redact_law_operator_inputs),
        "tail_loss_semantics": (
            "tail_law_signature_loss and tail_operator_seed_loss are aliases of the tail-read head losses; "
            "tail_operator_alignment_loss is the additional direct tail objective."
        ),
        "selection_objective": selection_mode,
        "selected_epoch": int(selected_epoch),
        "selected_score": float(selected_score),
        "final_epoch_score": float(final_epoch_score),
        "initial_batch_loss": float(initial_loss if initial_loss is not None else 0.0),
        "final_full_loss": final_loss,
        "final_matryoshka_loss": final_matryoshka_loss,
        "loss_reduction": float((initial_loss if initial_loss is not None else final_loss) - final_loss),
        "losses_by_epoch": losses_by_epoch,
        "aux_losses_by_epoch": aux_by_epoch,
        "final_metamer_consistency_loss": float(final_aux_losses["metamer_consistency_loss"].detach().cpu().item()),
        "final_case_separation_loss": float(final_aux_losses["case_separation_loss"].detach().cpu().item()),
        "final_mean_case_center_cosine": float(final_aux_losses["mean_case_center_cosine"].detach().cpu().item()),
        "final_case_separation_mean_case_center_cosine": float(
            final_aux_losses["case_separation_mean_case_center_cosine"].detach().cpu().item()
        ),
        "final_law_signature_loss": float(final_losses["law_signature_loss"].detach().cpu().item()),
        "final_operator_seed_loss": float(final_losses["operator_seed_loss"].detach().cpu().item()),
        "final_tail_law_signature_loss": float(final_losses["tail_law_signature_loss"].detach().cpu().item()),
        "final_tail_operator_seed_loss": float(final_losses["tail_operator_seed_loss"].detach().cpu().item()),
        "final_tail_operator_alignment_loss": float(
            final_losses["tail_operator_alignment_loss"].detach().cpu().item()
        ),
        "final_run_contrastive_loss": float(final_run_losses["run_contrastive_loss"].detach().cpu().item()),
        "final_run_positive_cosine": float(final_run_losses["run_positive_cosine"].detach().cpu().item()),
        "final_run_negative_cosine": float(final_run_losses["run_negative_cosine"].detach().cpu().item()),
        "run_center_count": int(final_run_losses["run_center_count"].detach().cpu().item()),
        "final_rank_entropy_loss": float(final_rank_losses["rank_entropy_loss"].detach().cpu().item()),
        "final_rank_entropy_effective_dim": float(final_rank_losses["rank_entropy_effective_dim"].detach().cpu().item()),
        "final_rank_entropy_fraction": float(final_rank_losses["rank_entropy_fraction"].detach().cpu().item()),
        "final_run_rank_entropy_loss": float(final_run_rank_losses["run_rank_entropy_loss"].detach().cpu().item()),
        "final_run_rank_effective_dim": float(final_run_rank_losses["run_rank_effective_dim"].detach().cpu().item()),
        "final_base_rank_entropy_loss": float(final_base_rank_losses["base_rank_entropy_loss"].detach().cpu().item()),
        "final_base_rank_effective_dim": float(final_base_rank_losses["base_rank_effective_dim"].detach().cpu().item()),
        "final_base_svd_rank_loss": float(final_base_svd_losses["base_svd_rank_loss"].detach().cpu().item()),
        "final_base_svd_rank_effective_dim": float(final_base_svd_losses["base_svd_rank_effective_dim"].detach().cpu().item()),
        "final_factor_geometry_loss": float(final_factor_geometry_losses["factor_geometry_loss"].detach().cpu().item()),
        "final_factor_geometry_cosine": float(final_factor_geometry_losses["factor_geometry_cosine"].detach().cpu().item()),
        "contract": contract,
    }


def _factor_geometry_losses(h: torch.Tensor, factor_rows: torch.Tensor) -> dict[str, torch.Tensor]:
    if h.numel() == 0 or factor_rows.numel() == 0 or h.size(0) != factor_rows.size(0) or h.size(0) < 2:
        zero = h.new_tensor(0.0)
        return {"factor_geometry_loss": zero, "factor_geometry_cosine": zero}
    learned = F.normalize(h.float().reshape(h.size(0), -1), dim=-1)
    target = F.normalize(factor_rows.float().reshape(factor_rows.size(0), -1), dim=-1)
    learned_sim = learned @ learned.t()
    target_sim = target @ target.t()
    mask = ~torch.eye(int(h.size(0)), dtype=torch.bool, device=h.device)
    learned_vec = learned_sim[mask]
    target_vec = target_sim[mask]
    loss = F.mse_loss(learned_vec, target_vec)
    cosine = F.cosine_similarity(learned_vec.flatten(), target_vec.flatten(), dim=0)
    return {"factor_geometry_loss": loss, "factor_geometry_cosine": cosine}


def _metamer_case_losses(
    h: torch.Tensor,
    *,
    packet_case_ids: torch.Tensor,
    separation_margin: float,
) -> dict[str, torch.Tensor]:
    if h.numel() == 0 or packet_case_ids.numel() != h.size(0):
        zero = h.new_tensor(0.0)
        return {
            "metamer_consistency_loss": zero,
            "case_separation_loss": zero,
            "mean_case_center_cosine": zero,
        }
    h_norm = F.normalize(h.float(), dim=-1)
    unique = torch.unique(packet_case_ids)
    centers = []
    consistency_terms = []
    for case_id in unique:
        mask = packet_case_ids == case_id
        rows = h_norm[mask]
        if rows.numel() == 0:
            continue
        center = F.normalize(rows.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        centers.append(center)
        consistency_terms.append(1.0 - F.cosine_similarity(rows, center.view(1, -1), dim=-1).mean())
    if not centers:
        zero = h.new_tensor(0.0)
        return {
            "metamer_consistency_loss": zero,
            "case_separation_loss": zero,
            "mean_case_center_cosine": zero,
        }
    consistency = torch.stack(consistency_terms).mean()
    center_t = torch.stack(centers, dim=0)
    if center_t.size(0) < 2:
        separation = h.new_tensor(0.0)
        mean_center_cos = h.new_tensor(1.0)
    else:
        sims = torch.mm(center_t, center_t.t()).clamp(-1.0, 1.0)
        tri = torch.triu_indices(sims.size(0), sims.size(1), offset=1, device=sims.device)
        vals = sims[tri[0], tri[1]]
        mean_center_cos = vals.mean()
        separation = F.relu(vals - float(separation_margin)).mean()
    return {
        "metamer_consistency_loss": consistency,
        "case_separation_loss": separation,
        "mean_case_center_cosine": mean_center_cos,
    }


def _metamer_case_losses_for_views(
    h: torch.Tensor,
    *,
    packet_case_ids: torch.Tensor,
    separation_margin: float,
    metamer_loss_view: str,
    case_separation_loss_view: str,
) -> dict[str, torch.Tensor]:
    metamer_losses = _metamer_case_losses(
        _signature_loss_view(h, metamer_loss_view),
        packet_case_ids=packet_case_ids,
        separation_margin=float(separation_margin),
    )
    separation_losses = _metamer_case_losses(
        _signature_loss_view(h, case_separation_loss_view),
        packet_case_ids=packet_case_ids,
        separation_margin=float(separation_margin),
    )
    merged = dict(metamer_losses)
    merged["case_separation_loss"] = separation_losses["case_separation_loss"]
    merged["case_separation_mean_case_center_cosine"] = separation_losses["mean_case_center_cosine"]
    return merged


def _run_center_contrastive_losses(
    h: torch.Tensor,
    *,
    packet_case_ids: torch.Tensor,
    packet_run_ids: torch.Tensor,
    temperature: float,
) -> dict[str, torch.Tensor]:
    if h.numel() == 0 or packet_case_ids.numel() != h.size(0) or packet_run_ids.numel() != h.size(0):
        zero = h.new_tensor(0.0)
        return {
            "run_contrastive_loss": zero,
            "run_positive_cosine": zero,
            "run_negative_cosine": zero,
            "run_center_count": zero,
        }
    h_norm = F.normalize(h.float(), dim=-1)
    centers = []
    labels = []
    for run_id in torch.unique(packet_run_ids):
        mask = packet_run_ids == run_id
        rows = h_norm[mask]
        if rows.numel() == 0:
            continue
        center = F.normalize(rows.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        centers.append(center)
        labels.append(packet_case_ids[mask][0])
    if len(centers) < 2:
        zero = h.new_tensor(0.0)
        return {
            "run_contrastive_loss": zero,
            "run_positive_cosine": h.new_tensor(1.0),
            "run_negative_cosine": zero,
            "run_center_count": h.new_tensor(float(len(centers))),
        }
    center_t = torch.stack(centers, dim=0)
    label_t = torch.stack(labels, dim=0)
    logits = torch.mm(center_t, center_t.t()).clamp(-1.0, 1.0) / max(float(temperature), 1.0e-4)
    eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    positive_mask = (label_t.view(-1, 1) == label_t.view(1, -1)) & ~eye
    negative_mask = (label_t.view(-1, 1) != label_t.view(1, -1))
    valid = positive_mask.any(dim=1)
    if not bool(valid.any().item()):
        zero = h.new_tensor(0.0)
        return {
            "run_contrastive_loss": zero,
            "run_positive_cosine": h.new_tensor(1.0),
            "run_negative_cosine": logits[negative_mask].mean() * max(float(temperature), 1.0e-4) if negative_mask.any() else zero,
            "run_center_count": h.new_tensor(float(len(centers))),
        }
    logits_no_self = logits.masked_fill(eye, -1.0e9)
    log_prob = logits_no_self - torch.logsumexp(logits_no_self, dim=1, keepdim=True)
    positive_log_prob = (log_prob * positive_mask.float()).sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0)
    loss = -positive_log_prob[valid].mean()
    sim = torch.mm(center_t, center_t.t()).clamp(-1.0, 1.0)
    positive_cos = sim[positive_mask].mean() if positive_mask.any() else h.new_tensor(1.0)
    negative_cos = sim[negative_mask].mean() if negative_mask.any() else h.new_tensor(0.0)
    return {
        "run_contrastive_loss": loss,
        "run_positive_cosine": positive_cos,
        "run_negative_cosine": negative_cos,
        "run_center_count": h.new_tensor(float(len(centers))),
    }


def _rank_entropy_losses(
    h: torch.Tensor,
    *,
    entropy_floor: float,
) -> dict[str, torch.Tensor]:
    if h.numel() == 0 or h.size(0) < 2:
        zero = h.new_tensor(0.0)
        return {
            "rank_entropy_loss": zero,
            "rank_entropy_effective_dim": zero,
            "rank_entropy_fraction": zero,
        }
    x = F.normalize(h.float(), dim=-1)
    centered = x - x.mean(dim=0, keepdim=True)
    var = centered.pow(2).mean(dim=0)
    probs = var / var.sum().clamp_min(1.0e-8)
    entropy = -(probs * probs.clamp_min(1.0e-8).log()).sum()
    effective_dim = torch.exp(entropy)
    fraction = effective_dim / h.new_tensor(float(h.size(-1)))
    loss = F.relu(h.new_tensor(float(entropy_floor)) - fraction)
    return {
        "rank_entropy_loss": loss,
        "rank_entropy_effective_dim": effective_dim,
        "rank_entropy_fraction": fraction,
    }


def _run_center_rank_entropy_losses(
    h: torch.Tensor,
    *,
    packet_run_ids: torch.Tensor,
    effective_dim_floor: float,
) -> dict[str, torch.Tensor]:
    if h.numel() == 0 or packet_run_ids.numel() != h.size(0):
        zero = h.new_tensor(0.0)
        return {
            "run_rank_entropy_loss": zero,
            "run_rank_effective_dim": zero,
        }
    h_norm = F.normalize(h.float(), dim=-1)
    centers = []
    for run_id in torch.unique(packet_run_ids):
        mask = packet_run_ids == run_id
        rows = h_norm[mask]
        if rows.numel() > 0:
            centers.append(F.normalize(rows.mean(dim=0, keepdim=True), dim=-1).squeeze(0))
    if len(centers) < 2:
        zero = h.new_tensor(0.0)
        return {
            "run_rank_entropy_loss": zero,
            "run_rank_effective_dim": h.new_tensor(float(len(centers))),
        }
    center_t = torch.stack(centers, dim=0)
    centered = center_t - center_t.mean(dim=0, keepdim=True)
    var = centered.pow(2).mean(dim=0)
    probs = var / var.sum().clamp_min(1.0e-8)
    entropy = -(probs * probs.clamp_min(1.0e-8).log()).sum()
    effective_dim = torch.exp(entropy)
    loss = F.relu(h.new_tensor(float(effective_dim_floor)) - effective_dim)
    return {
        "run_rank_entropy_loss": loss,
        "run_rank_effective_dim": effective_dim,
    }


def _effective_dim_floor_losses(
    h: torch.Tensor,
    *,
    effective_dim_floor: float,
    prefix: str,
) -> dict[str, torch.Tensor]:
    if h.numel() == 0 or h.size(0) < 2:
        zero = h.new_tensor(0.0)
        return {
            f"{prefix}_entropy_loss": zero,
            f"{prefix}_effective_dim": zero,
        }
    x = F.normalize(h.float(), dim=-1)
    centered = x - x.mean(dim=0, keepdim=True)
    var = centered.pow(2).mean(dim=0)
    probs = var / var.sum().clamp_min(1.0e-8)
    entropy = -(probs * probs.clamp_min(1.0e-8).log()).sum()
    effective_dim = torch.exp(entropy)
    loss = F.relu(h.new_tensor(float(effective_dim_floor)) - effective_dim)
    return {
        f"{prefix}_entropy_loss": loss,
        f"{prefix}_effective_dim": effective_dim,
    }


def _svd_effective_rank_losses(
    h: torch.Tensor,
    *,
    effective_rank_floor: float,
    prefix: str,
) -> dict[str, torch.Tensor]:
    if h.numel() == 0 or h.size(0) < 2:
        zero = h.new_tensor(0.0)
        return {
            f"{prefix}_loss": zero,
            f"{prefix}_effective_dim": zero,
        }
    x = F.normalize(h.float().reshape(h.size(0), -1), dim=-1)
    centered = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probs = singular / singular.sum().clamp_min(1.0e-8)
    entropy = -(probs * probs.clamp_min(1.0e-8).log()).sum()
    effective_rank = torch.exp(entropy)
    loss = F.relu(h.new_tensor(float(effective_rank_floor)) - effective_rank)
    return {
        f"{prefix}_loss": loss,
        f"{prefix}_effective_dim": effective_rank,
    }


def _evaluate_case(
    *,
    case_run: dict[str, Any],
    model: RafaRelationalSignatureEncoderV0,
    device: torch.device,
    cluster_threshold: float,
    prefix_dims: tuple[int, ...],
) -> dict[str, Any]:
    base_run = case_run["base_run"]
    base_pack = _factor_pack_for_run(base_run, device)
    learned_base = _learned_rows(model, base_pack, device)
    deterministic_base = _dense_rows(base_run).to(device)
    metadata_base, metadata_q_base, metadata_law_base = _metadata_rows(base_run)
    metadata_base = metadata_base.to(device)
    metadata_q_base = metadata_q_base.to(device)
    metadata_law_base = metadata_law_base.to(device)

    learned_pool = _pooled_rows(learned_base)
    learned_prefix640_pool = _pooled_rows(_signature_loss_view(learned_base, "prefix:640"))
    learned_tail640_pool = _pooled_rows(_signature_loss_view(learned_base, "tail:640"))
    deterministic_pool = _pooled_rows(deterministic_base)
    metadata_pool = _pooled_rows(metadata_base)
    metadata_q_pool = _pooled_rows(metadata_q_base)
    metadata_law_pool = _pooled_rows(metadata_law_base)

    transform_rows: list[dict[str, Any]] = []
    for transform_run in case_run["transform_runs"]:
        transform = transform_run["spec"]
        run_t = transform_run["run"]
        pack_t = _factor_pack_for_run(run_t, device)
        learned_t = _learned_rows(model, pack_t, device)
        deterministic_t = _dense_rows(run_t).to(device)
        metadata_t, metadata_q_t, metadata_law_t = _metadata_rows(run_t)
        metadata_t = metadata_t.to(device)
        metadata_q_t = metadata_q_t.to(device)
        metadata_law_t = metadata_law_t.to(device)

        learned_prefix = _prefix_metrics_from_rows(learned_base, learned_t, prefix_dims)
        deterministic_prefix = _dense_prefix_metrics(deterministic_base, deterministic_t, prefix_dims)
        learned_cos = _cos(learned_pool, _pooled_rows(learned_t))
        learned_prefix640_cos = _cos(learned_prefix640_pool, _pooled_rows(_signature_loss_view(learned_t, "prefix:640")))
        learned_tail640_cos = _cos(learned_tail640_pool, _pooled_rows(_signature_loss_view(learned_t, "tail:640")))
        deterministic_cos = _cos(deterministic_pool, _pooled_rows(deterministic_t))
        metadata_cos = _cos(metadata_pool, _pooled_rows(metadata_t))
        metadata_q_cos = _cos(metadata_q_pool, _pooled_rows(metadata_q_t))
        metadata_law_cos = _cos(metadata_law_pool, _pooled_rows(metadata_law_t))
        learned_stability = _mean([learned_cos, float(learned_prefix["mean_prefix_cos"])])
        deterministic_stability = _mean([deterministic_cos, float(deterministic_prefix["mean_prefix_cos"])])
        metadata_stability = _mean([metadata_cos, metadata_q_cos, metadata_law_cos])
        transform_rows.append(
            {
                "transform": str(transform["name"]),
                "transform_family": str(transform.get("family", transform.get("kind", "unknown"))),
                "learned": {
                    "pooled_full_cos": learned_cos,
                    "pooled_prefix640_cos": learned_prefix640_cos,
                    "pooled_tail640_cos": learned_tail640_cos,
                    "mean_prefix_cos": float(learned_prefix["mean_prefix_cos"]),
                    "prefix_cosines": learned_prefix["prefix_cosines"],
                    "stability_score": learned_stability,
                    "packet_count_delta": float(learned_t.size(0) - learned_base.size(0)),
                },
                "deterministic": {
                    "pooled_full_cos": deterministic_cos,
                    "mean_prefix_cos": float(deterministic_prefix["mean_prefix_cos"]),
                    "prefix_cosines": deterministic_prefix["prefix_cosines"],
                    "stability_score": deterministic_stability,
                    "packet_count_delta": float(deterministic_t.size(0) - deterministic_base.size(0)),
                },
                "metadata": {
                    "pooled_full_cos": metadata_cos,
                    "q_profile_cos": metadata_q_cos,
                    "law_signature_cos": metadata_law_cos,
                    "stability_score": metadata_stability,
                    "packet_count_delta": float(metadata_t.size(0) - metadata_base.size(0)),
                },
                "delta_learned_minus_metadata_stability": learned_stability - metadata_stability,
                "delta_learned_minus_deterministic_stability": learned_stability - deterministic_stability,
            }
        )

    learned_stability_values = [float(row["learned"]["stability_score"]) for row in transform_rows]
    learned_prefix640_stability_values = [float(row["learned"]["pooled_prefix640_cos"]) for row in transform_rows]
    learned_tail640_stability_values = [float(row["learned"]["pooled_tail640_cos"]) for row in transform_rows]
    deterministic_stability_values = [float(row["deterministic"]["stability_score"]) for row in transform_rows]
    metadata_stability_values = [float(row["metadata"]["stability_score"]) for row in transform_rows]
    return {
        "case": case_run["case"],
        "wav_path": case_run["wav_path"],
        "base": {
            "num_learned_signatures": int(learned_base.size(0)),
            "num_deterministic_signatures": int(deterministic_base.size(0)),
            "num_metadata_packets": int(metadata_base.size(0)),
            "learned": _representation_summary(learned_base.detach().cpu(), threshold=cluster_threshold),
            "deterministic": _representation_summary(deterministic_base.detach().cpu(), threshold=cluster_threshold),
            "metadata": _representation_summary(metadata_base.detach().cpu(), threshold=cluster_threshold),
            "learned_operator_tail": _representation_summary(
                learned_base[:, 640:768].detach().cpu() if learned_base.numel() and learned_base.size(-1) >= 768 else _empty_rows(),
                threshold=cluster_threshold,
            ),
        },
        "stability": {
            "learned_mean_stability_score": _mean(learned_stability_values),
            "deterministic_mean_stability_score": _mean(deterministic_stability_values),
            "metadata_mean_stability_score": _mean(metadata_stability_values),
            "delta_learned_minus_metadata": _mean(learned_stability_values) - _mean(metadata_stability_values),
            "delta_learned_minus_deterministic": _mean(learned_stability_values) - _mean(deterministic_stability_values),
            "learned_std_stability_score": _stddev(learned_stability_values),
            "deterministic_std_stability_score": _stddev(deterministic_stability_values),
            "metadata_std_stability_score": _stddev(metadata_stability_values),
        },
        "slice_stability": {
            "prefix640_learned_mean_score": _mean(learned_prefix640_stability_values),
            "tail640_learned_mean_score": _mean(learned_tail640_stability_values),
            "prefix640_learned_std_score": _stddev(learned_prefix640_stability_values),
            "tail640_learned_std_score": _stddev(learned_tail640_stability_values),
        },
        "transforms": transform_rows,
        "_learned_base_pool": learned_pool.detach().cpu() if learned_pool is not None else None,
        "_learned_prefix640_base_pool": learned_prefix640_pool.detach().cpu() if learned_prefix640_pool is not None else None,
        "_learned_tail640_base_pool": learned_tail640_pool.detach().cpu() if learned_tail640_pool is not None else None,
        "_deterministic_base_pool": deterministic_pool.detach().cpu() if deterministic_pool is not None else None,
        "_metadata_base_pool": metadata_pool.detach().cpu() if metadata_pool is not None else None,
        "_learned_packet_rows": learned_base.detach().cpu(),
        "_learned_prefix640_packet_rows": _signature_loss_view(learned_base, "prefix:640").detach().cpu(),
        "_learned_tail640_packet_rows": _signature_loss_view(learned_base, "tail:640").detach().cpu(),
        "_deterministic_packet_rows": deterministic_base.detach().cpu(),
        "_metadata_packet_rows": metadata_base.detach().cpu(),
    }


def _strip_case(case: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if not key.startswith("_")}


def _stack_optional(rows: list[torch.Tensor | None]) -> torch.Tensor:
    kept = [row.float().view(-1) for row in rows if row is not None and row.numel() > 0]
    if not kept:
        return _empty_rows()
    min_dim = min(int(row.numel()) for row in kept)
    return torch.stack([row[:min_dim] for row in kept], dim=0)


def _aggregate_cases(cases: list[dict[str, Any]], cluster_threshold: float) -> dict[str, Any]:
    learned_case_rows = _stack_optional([case.get("_learned_base_pool") for case in cases])
    learned_prefix640_case_rows = _stack_optional([case.get("_learned_prefix640_base_pool") for case in cases])
    learned_tail640_case_rows = _stack_optional([case.get("_learned_tail640_base_pool") for case in cases])
    deterministic_case_rows = _stack_optional([case.get("_deterministic_base_pool") for case in cases])
    metadata_case_rows = _stack_optional([case.get("_metadata_base_pool") for case in cases])
    learned_packet_rows = _concat_rows([case["_learned_packet_rows"] for case in cases])
    learned_prefix640_packet_rows = _concat_rows([case["_learned_prefix640_packet_rows"] for case in cases])
    learned_tail640_packet_rows = _concat_rows([case["_learned_tail640_packet_rows"] for case in cases])
    deterministic_packet_rows = _concat_rows([case["_deterministic_packet_rows"] for case in cases])
    metadata_packet_rows = _concat_rows([case["_metadata_packet_rows"] for case in cases])

    learned_stability = [float(case["stability"]["learned_mean_stability_score"]) for case in cases]
    deterministic_stability = [float(case["stability"]["deterministic_mean_stability_score"]) for case in cases]
    metadata_stability = [float(case["stability"]["metadata_mean_stability_score"]) for case in cases]
    learned_sep = _pairwise_summary(learned_case_rows, near_duplicate_threshold=cluster_threshold)
    learned_prefix640_sep = _pairwise_summary(learned_prefix640_case_rows, near_duplicate_threshold=cluster_threshold)
    learned_tail640_sep = _pairwise_summary(learned_tail640_case_rows, near_duplicate_threshold=cluster_threshold)
    deterministic_sep = _pairwise_summary(deterministic_case_rows, near_duplicate_threshold=cluster_threshold)
    metadata_sep = _pairwise_summary(metadata_case_rows, near_duplicate_threshold=cluster_threshold)
    learned_collapse = _representation_summary(learned_packet_rows, threshold=cluster_threshold)
    learned_prefix640_collapse = _representation_summary(learned_prefix640_packet_rows, threshold=cluster_threshold)
    learned_tail640_collapse = _representation_summary(learned_tail640_packet_rows, threshold=cluster_threshold)
    deterministic_collapse = _representation_summary(deterministic_packet_rows, threshold=cluster_threshold)
    metadata_collapse = _representation_summary(metadata_packet_rows, threshold=cluster_threshold)
    return {
        "num_cases": len(cases),
        "stability": {
            "learned_score": _mean(learned_stability),
            "deterministic_score": _mean(deterministic_stability),
            "metadata_score": _mean(metadata_stability),
            "delta_learned_minus_metadata": _mean(learned_stability) - _mean(metadata_stability),
            "delta_learned_minus_deterministic": _mean(learned_stability) - _mean(deterministic_stability),
            "learned_std": _stddev(learned_stability),
            "deterministic_std": _stddev(deterministic_stability),
            "metadata_std": _stddev(metadata_stability),
        },
        "separation": {
            "learned_score": float(learned_sep["mean_pairwise_distance"]),
            "deterministic_score": float(deterministic_sep["mean_pairwise_distance"]),
            "metadata_score": float(metadata_sep["mean_pairwise_distance"]),
            "delta_learned_minus_metadata": float(learned_sep["mean_pairwise_distance"] - metadata_sep["mean_pairwise_distance"]),
            "delta_learned_minus_deterministic": float(learned_sep["mean_pairwise_distance"] - deterministic_sep["mean_pairwise_distance"]),
            "learned": learned_sep,
            "deterministic": deterministic_sep,
            "metadata": metadata_sep,
        },
        "collapse": {
            "learned": learned_collapse,
            "deterministic": deterministic_collapse,
            "metadata": metadata_collapse,
            "delta_learned_minus_metadata": {
                "dominant_cluster_share": float(learned_collapse["dominant_cluster_share"] - metadata_collapse["dominant_cluster_share"]),
                "near_duplicate_pair_share": float(learned_collapse["near_duplicate_pair_share"] - metadata_collapse["near_duplicate_pair_share"]),
                "effective_rank": float(learned_collapse["effective_rank"] - metadata_collapse["effective_rank"]),
                "mean_pairwise_distance": float(learned_collapse["mean_pairwise_distance"] - metadata_collapse["mean_pairwise_distance"]),
                "effective_cluster_count": float(learned_collapse["effective_cluster_count"] - metadata_collapse["effective_cluster_count"]),
            },
            "delta_learned_minus_deterministic": {
                "dominant_cluster_share": float(learned_collapse["dominant_cluster_share"] - deterministic_collapse["dominant_cluster_share"]),
                "near_duplicate_pair_share": float(learned_collapse["near_duplicate_pair_share"] - deterministic_collapse["near_duplicate_pair_share"]),
                "effective_rank": float(learned_collapse["effective_rank"] - deterministic_collapse["effective_rank"]),
                "mean_pairwise_distance": float(learned_collapse["mean_pairwise_distance"] - deterministic_collapse["mean_pairwise_distance"]),
                "effective_cluster_count": float(learned_collapse["effective_cluster_count"] - deterministic_collapse["effective_cluster_count"]),
            },
        },
        "operator_tail": _representation_summary(
            learned_packet_rows[:, 640:768] if learned_packet_rows.numel() and learned_packet_rows.size(-1) >= 768 else _empty_rows(),
            threshold=cluster_threshold,
        ),
        "slice_views": {
            "prefix640": {
                "stability_score": _mean(
                    [float(case.get("slice_stability", {}).get("prefix640_learned_mean_score", 0.0)) for case in cases]
                ),
                "separation": learned_prefix640_sep,
                "collapse": learned_prefix640_collapse,
            },
            "tail640": {
                "stability_score": _mean(
                    [float(case.get("slice_stability", {}).get("tail640_learned_mean_score", 0.0)) for case in cases]
                ),
                "separation": learned_tail640_sep,
                "collapse": learned_tail640_collapse,
            },
        },
    }


def _verdict(aggregate: dict[str, Any], *, split_mode: str) -> dict[str, Any]:
    stability_delta = _as_float(aggregate["stability"]["delta_learned_minus_metadata"])
    separation_delta = _as_float(aggregate["separation"]["delta_learned_minus_metadata"])
    collapse_delta = aggregate["collapse"]["delta_learned_minus_metadata"]
    learned_beats_metadata_stability = stability_delta > 1.0e-4
    learned_beats_metadata_separation = separation_delta > 1.0e-4
    learned_no_worse_collapse = (
        _as_float(collapse_delta.get("dominant_cluster_share")) <= 0.02
        and _as_float(collapse_delta.get("near_duplicate_pair_share")) <= 0.02
        and _as_float(collapse_delta.get("effective_rank")) >= -0.05
        and _as_float(collapse_delta.get("mean_pairwise_distance")) >= -0.02
    )
    reasons: list[str] = []
    if split_mode == "in_sample":
        reasons.append("Scout is in-sample; use only as implementation signal, not promotion evidence.")
    if not learned_beats_metadata_stability:
        reasons.append("Learned h does not beat explicit metadata on metamer stability.")
    if not learned_beats_metadata_separation:
        reasons.append("Learned h does not beat explicit metadata on case separation.")
    if not learned_no_worse_collapse:
        reasons.append("Learned h has worse anti-collapse diagnostics than explicit metadata.")
    all_axes = learned_beats_metadata_stability and learned_beats_metadata_separation and learned_no_worse_collapse
    if all_axes and str(split_mode).startswith("heldout"):
        status = "learned_signature_candidate_holdout_passed_needs_seed"
    elif all_axes:
        status = "learned_signature_candidate_needs_holdout"
    else:
        status = "learned_signature_scout_not_candidate"
    return {
        "status": status,
        "promotion_effect": "none",
        "split_mode": split_mode,
        "criteria": {
            "learned_beats_metadata_stability": learned_beats_metadata_stability,
            "learned_beats_metadata_separation": learned_beats_metadata_separation,
            "learned_no_worse_collapse": learned_no_worse_collapse,
            "all_axes": all_axes,
        },
        "reasons": reasons,
    }


def train_learned_signature_scout(
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
    metamer_loss_view: str,
    case_separation_loss_view: str,
    run_contrastive_loss_view: str,
    redact_law_operator_inputs: bool,
    seed: int,
    case_offset: int = 0,
    case_shuffle_seed: int | None = None,
    selection_objective: str = "final",
) -> dict[str, Any]:
    device = _safe_device(device_name)
    _set_rng_seed(int(seed))
    cfg = _load_circle_cfg(config_path)
    stft_cfg = load_config()["data"]["stft"]
    all_cases = _rotate_cases(_shuffle_cases(_load_cases(cases_path), case_shuffle_seed), int(case_offset))
    train_case_count = int(num_cases) if int(num_cases) > 0 else len(all_cases)
    selected_cases = _select_case_slice(all_cases, start=0, count=train_case_count)
    heldout_selected_cases = (
        _select_case_slice(all_cases, start=len(selected_cases), count=int(heldout_cases))
        if int(heldout_cases) > 0
        else {}
    )
    transforms = _transform_specs()[: max(0, int(num_transforms))]
    cluster_threshold = float(getattr(cfg, "law_packet_merge_threshold", DEFAULT_CLUSTER_THRESHOLD))
    out_dir.mkdir(parents=True, exist_ok=True)

    case_runs: list[dict[str, Any]] = []
    heldout_case_runs: list[dict[str, Any]] = []
    packs: list[RelationalFactorPack] = []
    packet_case_ids: list[int] = []
    packet_run_ids: list[int] = []
    packet_is_base: list[bool] = []
    next_run_id = 0
    for case_idx, (case_name, wav_path_str) in enumerate(selected_cases.items()):
        row = _case_runs(
            case_name=case_name,
            wav_path=Path(wav_path_str),
            cfg=cfg,
            stft_cfg=stft_cfg,
            device=device,
            device_name=device_name,
            clip_seconds=int(clip_seconds),
            transforms=transforms,
        )
        case_runs.append(row)
        base_pack = _factor_pack_for_run(row["base_run"], device)
        if base_pack is not None:
            packs.append(base_pack)
            packet_case_ids.extend([case_idx] * int(base_pack.q_profile.size(0)))
            packet_run_ids.extend([next_run_id] * int(base_pack.q_profile.size(0)))
            packet_is_base.extend([True] * int(base_pack.q_profile.size(0)))
            next_run_id += 1
        for transform_run in row["transform_runs"]:
            pack_t = _factor_pack_for_run(transform_run["run"], device)
            if pack_t is not None:
                packs.append(pack_t)
                packet_case_ids.extend([case_idx] * int(pack_t.q_profile.size(0)))
                packet_run_ids.extend([next_run_id] * int(pack_t.q_profile.size(0)))
                packet_is_base.extend([False] * int(pack_t.q_profile.size(0)))
                next_run_id += 1
    for case_name, wav_path_str in heldout_selected_cases.items():
        heldout_case_runs.append(
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

    train_pack = _concat_factor_packs(packs)
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
        metamer_loss_view=str(metamer_loss_view),
        case_separation_loss_view=str(case_separation_loss_view),
        run_contrastive_loss_view=str(run_contrastive_loss_view),
        redact_law_operator_inputs=bool(redact_law_operator_inputs),
        selection_objective=str(selection_objective),
    )
    train_summary["seed"] = int(seed)
    prefix_dims = tuple(int(dim) for dim in model.cfg.prefix_dims)
    train_case_summaries_internal = [
        _evaluate_case(
            case_run=row,
            model=model,
            device=device,
            cluster_threshold=cluster_threshold,
            prefix_dims=prefix_dims,
        )
        for row in case_runs
    ]
    heldout_case_summaries_internal = [
        _evaluate_case(
            case_run=row,
            model=model,
            device=device,
            cluster_threshold=cluster_threshold,
            prefix_dims=prefix_dims,
        )
        for row in heldout_case_runs
    ]
    train_aggregate = _aggregate_cases(train_case_summaries_internal, cluster_threshold)
    heldout_aggregate = (
        _aggregate_cases(heldout_case_summaries_internal, cluster_threshold)
        if heldout_case_summaries_internal
        else None
    )
    aggregate = heldout_aggregate if heldout_aggregate is not None else train_aggregate
    case_summaries_internal = heldout_case_summaries_internal if heldout_case_summaries_internal else train_case_summaries_internal
    split_mode = "heldout_cases" if heldout_case_summaries_internal else "in_sample"
    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "runtime": "circleworld_proto",
        "config_path": str(config_path),
        "cases_path": str(cases_path),
        "out_dir": str(out_dir),
        "device": str(device),
        "requested_device": str(device_name),
        "num_cases": len(case_summaries_internal),
        "train_case_count": len(train_case_summaries_internal),
        "heldout_case_count": len(heldout_case_summaries_internal),
        "num_transforms": len(transforms),
        "clip_seconds": int(clip_seconds),
        "seed": int(seed),
        "case_offset": int(case_offset),
        "case_shuffle_seed": None if case_shuffle_seed is None else int(case_shuffle_seed),
        "cluster_threshold": cluster_threshold,
        "split_mode": split_mode,
        "train_case_names": list(selected_cases.keys()),
        "heldout_case_names": list(heldout_selected_cases.keys()),
        "train_summary": train_summary,
        "aggregate": aggregate,
        "train_aggregate": train_aggregate,
        "heldout_aggregate": heldout_aggregate,
        "train_cases": [_strip_case(case) for case in train_case_summaries_internal],
        "heldout_cases": [_strip_case(case) for case in heldout_case_summaries_internal],
        "cases": [_strip_case(case) for case in case_summaries_internal],
    }
    summary["verdict"] = _verdict(aggregate, split_mode=split_mode)
    json_path = out_dir / "learned_signature_scout.json"
    md_path = out_dir / "LEARNED_SIGNATURE_SCOUT.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    train = summary["train_summary"]
    verdict = summary["verdict"]
    lines = [
        "# Learned Signature Scout",
        "",
        f"- schema: `{summary['schema']}`",
        f"- status: `{verdict['status']}`",
        f"- promotion effect: `{verdict['promotion_effect']}`",
        f"- split mode: `{summary['split_mode']}`",
        f"- cases: `{summary['num_cases']}`",
        f"- train cases: `{summary.get('train_case_count')}`",
        f"- heldout cases: `{summary.get('heldout_case_count')}`",
        f"- transforms: `{summary['num_transforms']}`",
        f"- case offset: `{summary.get('case_offset', 0)}`",
        f"- case shuffle seed: `{summary.get('case_shuffle_seed')}`",
        f"- train packets: `{train['train_packet_count']}`",
        f"- seed: `{train.get('seed')}`",
        f"- metamer consistency weight: `{train['metamer_consistency_weight']}`",
        f"- case separation weight: `{train['case_separation_weight']}`",
        f"- run contrastive weight: `{train['run_contrastive_weight']}`",
        f"- run contrastive temperature: `{train['run_contrastive_temperature']}`",
        f"- rank entropy weight: `{train['rank_entropy_weight']}`",
        f"- rank entropy floor: `{train['rank_entropy_floor']}`",
        f"- run rank entropy weight: `{train['run_rank_entropy_weight']}`",
        f"- run rank effective dim floor: `{train['run_rank_effective_dim_floor']}`",
        f"- base rank entropy weight: `{train['base_rank_entropy_weight']}`",
        f"- base rank effective dim floor: `{train['base_rank_effective_dim_floor']}`",
        f"- base SVD rank weight: `{train['base_svd_rank_weight']}`",
        f"- base SVD rank floor: `{train['base_svd_rank_floor']}`",
        f"- factor geometry weight: `{train.get('factor_geometry_weight')}`",
        f"- metamer consistency slice: `{train.get('metamer_consistency_slice', train.get('metamer_loss_view', 'full'))}`",
        f"- case separation slice: `{train.get('case_separation_slice', train.get('case_separation_loss_view', 'full'))}`",
        f"- run contrastive slice: `{train.get('run_contrastive_slice', train.get('run_contrastive_loss_view', 'full'))}`",
        f"- redact law/operator inputs: `{train.get('redact_law_operator_inputs')}`",
        f"- tail loss semantics: {train.get('tail_loss_semantics')}",
        f"- selection objective: `{train.get('selection_objective')}`",
        f"- selected epoch: `{train.get('selected_epoch')}`",
        f"- selected score: `{train.get('selected_score', 0.0):.6f}`",
        f"- final epoch score: `{train.get('final_epoch_score', 0.0):.6f}`",
        f"- initial batch loss: `{train['initial_batch_loss']:.6f}`",
        f"- final full loss: `{train['final_full_loss']:.6f}`",
        f"- loss reduction: `{train['loss_reduction']:.6f}`",
        f"- final run positive cosine: `{train['final_run_positive_cosine']:.6f}`",
        f"- final run negative cosine: `{train['final_run_negative_cosine']:.6f}`",
        f"- final rank entropy effective dim: `{train['final_rank_entropy_effective_dim']:.6f}`",
        f"- final rank entropy fraction: `{train['final_rank_entropy_fraction']:.6f}`",
        f"- final run rank effective dim: `{train['final_run_rank_effective_dim']:.6f}`",
        f"- final base rank effective dim: `{train['final_base_rank_effective_dim']:.6f}`",
        f"- final base SVD rank effective dim: `{train['final_base_svd_rank_effective_dim']:.6f}`",
        f"- final factor geometry cosine: `{train.get('final_factor_geometry_cosine', 0.0):.6f}`",
        f"- final law signature loss: `{train.get('final_law_signature_loss', 0.0):.6f}`",
        f"- final operator seed loss: `{train.get('final_operator_seed_loss', 0.0):.6f}`",
        f"- final tail law signature loss: `{train.get('final_tail_law_signature_loss', 0.0):.6f}`",
        f"- final tail operator seed loss: `{train.get('final_tail_operator_seed_loss', 0.0):.6f}`",
        f"- final tail operator alignment loss: `{train.get('final_tail_operator_alignment_loss', 0.0):.6f}`",
        "",
        "## Aggregate",
        "",
        f"- learned stability: `{agg['stability']['learned_score']:.6f}`",
        f"- deterministic stability: `{agg['stability']['deterministic_score']:.6f}`",
        f"- metadata stability: `{agg['stability']['metadata_score']:.6f}`",
        f"- learned-minus-metadata stability: `{agg['stability']['delta_learned_minus_metadata']:+.6f}`",
        f"- learned-minus-deterministic stability: `{agg['stability']['delta_learned_minus_deterministic']:+.6f}`",
        f"- learned separation: `{agg['separation']['learned_score']:.6f}`",
        f"- deterministic separation: `{agg['separation']['deterministic_score']:.6f}`",
        f"- metadata separation: `{agg['separation']['metadata_score']:.6f}`",
        f"- learned-minus-metadata separation: `{agg['separation']['delta_learned_minus_metadata']:+.6f}`",
        f"- learned dominant cluster share: `{agg['collapse']['learned']['dominant_cluster_share']:.6f}`",
        f"- learned near-duplicate pair share: `{agg['collapse']['learned']['near_duplicate_pair_share']:.6f}`",
        f"- learned operator-tail effective rank: `{agg['operator_tail']['effective_rank']:.6f}`",
        "",
        "## Verdict Reasons",
        "",
    ]
    lines.extend([f"- {reason}" for reason in verdict.get("reasons", [])] or ["- No verdict reason recorded."])
    lines.extend(["", "## Per Case", ""])
    for case in summary.get("cases", []):
        stability = case["stability"]
        lines.extend(
            [
                f"### {case['case']}",
                f"- learned stability: `{stability['learned_mean_stability_score']:.6f}`",
                f"- deterministic stability: `{stability['deterministic_mean_stability_score']:.6f}`",
                f"- metadata stability: `{stability['metadata_mean_stability_score']:.6f}`",
                f"- learned-minus-metadata: `{stability['delta_learned_minus_metadata']:+.6f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a small learned RAFA signature encoder scout on actual Circleworld packet factors.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--cases-json", default=str(_default_cases_for_scout()))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=6)
    ap.add_argument("--heldout-cases", type=int, default=0)
    ap.add_argument("--num-transforms", type=int, default=4)
    ap.add_argument("--clip-seconds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--learning-rate", type=float, default=3.0e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--metamer-consistency-weight", type=float, default=0.0)
    ap.add_argument("--case-separation-weight", type=float, default=0.0)
    ap.add_argument("--separation-margin", type=float, default=0.65)
    ap.add_argument("--run-contrastive-weight", type=float, default=0.0)
    ap.add_argument("--run-contrastive-temperature", type=float, default=0.12)
    ap.add_argument("--rank-entropy-weight", type=float, default=0.0)
    ap.add_argument("--rank-entropy-floor", type=float, default=0.02)
    ap.add_argument("--run-rank-entropy-weight", type=float, default=0.0)
    ap.add_argument("--run-rank-effective-dim-floor", type=float, default=3.0)
    ap.add_argument("--base-rank-entropy-weight", type=float, default=0.0)
    ap.add_argument("--base-rank-effective-dim-floor", type=float, default=3.0)
    ap.add_argument("--base-svd-rank-weight", type=float, default=0.0)
    ap.add_argument("--base-svd-rank-floor", type=float, default=2.5)
    ap.add_argument("--factor-geometry-weight", type=float, default=0.0)
    ap.add_argument("--metamer-consistency-slice", default="full")
    ap.add_argument("--case-separation-slice", default="full")
    ap.add_argument("--run-contrastive-slice", default="full")
    ap.add_argument("--redact-law-operator-inputs", action="store_true")
    ap.add_argument("--selection-objective", choices=("final", "train_proxy"), default="final")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--case-offset", type=int, default=0)
    ap.add_argument("--case-shuffle-seed", type=int, default=None)
    args = ap.parse_args()
    summary = train_learned_signature_scout(
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
        metamer_loss_view=str(args.metamer_consistency_slice),
        case_separation_loss_view=str(args.case_separation_slice),
        run_contrastive_loss_view=str(args.run_contrastive_slice),
        redact_law_operator_inputs=bool(args.redact_law_operator_inputs),
        selection_objective=str(args.selection_objective),
        seed=int(args.seed),
        case_offset=int(args.case_offset),
        case_shuffle_seed=args.case_shuffle_seed,
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "saved": str(Path(args.out_dir) / "learned_signature_scout.json"),
                "report": str(Path(args.out_dir) / "LEARNED_SIGNATURE_SCOUT.md"),
                "status": summary["verdict"]["status"],
                "promotion_effect": summary["verdict"]["promotion_effect"],
                "split_mode": summary["split_mode"],
                "seed": summary["seed"],
                "case_offset": summary.get("case_offset", 0),
                "case_shuffle_seed": summary.get("case_shuffle_seed"),
                "train_case_count": summary["train_case_count"],
                "heldout_case_count": summary["heldout_case_count"],
                "aggregate": {
                    "learned_minus_metadata_stability": summary["aggregate"]["stability"]["delta_learned_minus_metadata"],
                    "learned_minus_metadata_separation": summary["aggregate"]["separation"]["delta_learned_minus_metadata"],
                    "learned_dominant_cluster_share": summary["aggregate"]["collapse"]["learned"]["dominant_cluster_share"],
                    "learned_near_duplicate_pair_share": summary["aggregate"]["collapse"]["learned"]["near_duplicate_pair_share"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
