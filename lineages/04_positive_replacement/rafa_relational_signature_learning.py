from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rafa_relational_signature import (
    RafaRelationalSignatureConfig,
    RelationalFactorPack,
)


EPS = 1.0e-8


@dataclass
class RafaRelationalSignatureLearningConfig:
    signature_dim: int = 768
    prefix_dims: tuple[int, ...] = (128, 256, 384, 512, 640, 768)
    hidden_dim: int = 512
    coarse_dim: int = 24
    q_dim: int = 7
    arc_dim: int = 5
    temporal_dim: int = 8
    support_dim: int = 6
    num_modes: int = 2
    branch_profile_dim: int = 4
    law_signature_dim: int = 18
    operator_seed_dim: int = 128
    dropout: float = 0.0
    redact_law_operator_input: bool = False

    @property
    def branch_flat_dim(self) -> int:
        return int(self.num_modes) * int(self.branch_profile_dim)

    @property
    def input_dim(self) -> int:
        return (
            int(self.coarse_dim)
            + int(self.q_dim)
            + int(self.arc_dim)
            + int(self.temporal_dim)
            + int(self.support_dim)
            + int(self.branch_flat_dim)
            + int(self.law_signature_dim)
            + 1
            + int(self.operator_seed_dim)
        )

    @classmethod
    def from_factor_pack(
        cls,
        pack: RelationalFactorPack,
        *,
        signature_dim: int = 768,
        prefix_dims: tuple[int, ...] = (128, 256, 384, 512, 640, 768),
        hidden_dim: int = 512,
        dropout: float = 0.0,
        redact_law_operator_input: bool = False,
    ) -> RafaRelationalSignatureLearningConfig:
        return cls(
            signature_dim=int(signature_dim),
            prefix_dims=tuple(int(dim) for dim in prefix_dims),
            hidden_dim=int(hidden_dim),
            coarse_dim=int(pack.coarse_profile.size(-1)),
            q_dim=int(pack.q_profile.size(-1)),
            arc_dim=int(pack.arc_profile.size(-1)),
            temporal_dim=int(pack.temporal_profile.size(-1)),
            support_dim=int(pack.support_profile.size(-1)),
            num_modes=int(pack.branch_profile.size(1)),
            branch_profile_dim=int(pack.branch_profile.size(-1)),
            law_signature_dim=int(pack.law_signature.size(-1)),
            operator_seed_dim=int(pack.operator_seed.size(-1)),
            dropout=float(dropout),
            redact_law_operator_input=bool(redact_law_operator_input),
        )

    def signature_config(self) -> RafaRelationalSignatureConfig:
        return RafaRelationalSignatureConfig(
            signature_dim=int(self.signature_dim),
            prefix_dims=tuple(int(dim) for dim in self.prefix_dims),
            operator_seed_dim=int(self.operator_seed_dim),
            branch_profile_dim=int(self.branch_profile_dim),
        )


@dataclass
class RafaRelationalSignatureHeadOutputs:
    h: torch.Tensor
    q_profile: torch.Tensor
    arc_profile: torch.Tensor
    temporal_profile: torch.Tensor
    support_profile: torch.Tensor
    branch_profile: torch.Tensor
    law_signature: torch.Tensor
    operator_seed: torch.Tensor
    confidence: torch.Tensor
    redact_law_operator_input: bool = False


@dataclass
class RafaSignatureLossWeights:
    q_profile: float = 1.0
    arc_profile: float = 0.5
    temporal_profile: float = 0.75
    support_profile: float = 0.75
    branch_profile: float = 0.75
    law_signature: float = 0.75
    operator_seed: float = 1.0
    tail_operator_alignment: float = 0.25
    confidence: float = 0.25
    unit_norm: float = 0.1


def _prefix(h: torch.Tensor, dim: int) -> torch.Tensor:
    use_dim = max(1, min(int(dim), int(h.size(-1))))
    return h[:, :use_dim]


def _tail(h: torch.Tensor, start: int, end: int) -> torch.Tensor:
    start_idx = max(0, min(int(start), int(h.size(-1)) - 1))
    end_idx = max(start_idx + 1, min(int(end), int(h.size(-1))))
    return h[:, start_idx:end_idx]


def _normalized_rows(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return F.normalize(x, dim=-1)


def factor_pack_to_learning_input(
    pack: RelationalFactorPack,
    cfg: RafaRelationalSignatureLearningConfig | None = None,
) -> torch.Tensor:
    cfg = cfg or RafaRelationalSignatureLearningConfig.from_factor_pack(pack)
    pack = pack.validate(cfg.signature_config())
    branch_flat = pack.branch_profile.reshape(pack.branch_profile.size(0), -1)
    parts = [
        pack.coarse_profile,
        pack.q_profile,
        pack.arc_profile,
        pack.temporal_profile,
        pack.support_profile,
        branch_flat,
        torch.zeros_like(pack.law_signature) if cfg.redact_law_operator_input else pack.law_signature,
        pack.confidence,
        torch.zeros_like(pack.operator_seed) if cfg.redact_law_operator_input else pack.operator_seed,
    ]
    x = torch.cat([part.float() for part in parts], dim=-1)
    if int(x.size(-1)) != int(cfg.input_dim):
        raise ValueError(f"learning input dim mismatch: expected {cfg.input_dim}, got {int(x.size(-1))}")
    return x


class RafaRelationalSignatureProjectionHeadsV0(nn.Module):
    def __init__(self, cfg: RafaRelationalSignatureLearningConfig):
        super().__init__()
        self.cfg = cfg
        p128, p256, p384, p512, p640, p768 = [int(dim) for dim in cfg.prefix_dims]
        self.confidence_head = nn.Linear(p128, 1)
        self.q_head = nn.Linear(p256, int(cfg.q_dim))
        self.arc_head = nn.Linear(p256, int(cfg.arc_dim))
        self.temporal_head = nn.Linear(p384, int(cfg.temporal_dim))
        self.support_head = nn.Linear(p512, int(cfg.support_dim))
        self.branch_head = nn.Linear(p640, int(cfg.branch_flat_dim))
        tail_dim = max(1, p768 - p640)
        self.law_head = nn.Linear(tail_dim, int(cfg.law_signature_dim))
        self.operator_head = nn.Linear(tail_dim, int(cfg.operator_seed_dim))

    def forward(self, h: torch.Tensor) -> RafaRelationalSignatureHeadOutputs:
        cfg = self.cfg
        operator_tail = _tail(h, cfg.prefix_dims[4], cfg.prefix_dims[5])
        return RafaRelationalSignatureHeadOutputs(
            h=h,
            q_profile=F.softmax(self.q_head(_prefix(h, cfg.prefix_dims[1])), dim=-1),
            arc_profile=torch.sigmoid(self.arc_head(_prefix(h, cfg.prefix_dims[1]))),
            temporal_profile=torch.sigmoid(self.temporal_head(_prefix(h, cfg.prefix_dims[2]))),
            support_profile=torch.sigmoid(self.support_head(_prefix(h, cfg.prefix_dims[3]))),
            branch_profile=torch.sigmoid(self.branch_head(_prefix(h, cfg.prefix_dims[4]))).view(
                h.size(0),
                int(cfg.num_modes),
                int(cfg.branch_profile_dim),
            ),
            law_signature=_normalized_rows(self.law_head(operator_tail)),
            operator_seed=_normalized_rows(self.operator_head(operator_tail)),
            confidence=torch.sigmoid(self.confidence_head(_prefix(h, cfg.prefix_dims[0]))),
            redact_law_operator_input=bool(cfg.redact_law_operator_input),
        )


class RafaRelationalSignatureEncoderV0(nn.Module):
    """Learned dense-body prototype; not wired into Circleworld runtime by default."""

    def __init__(self, cfg: RafaRelationalSignatureLearningConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(int(cfg.input_dim), int(cfg.hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.signature_dim)),
        )
        self.heads = RafaRelationalSignatureProjectionHeadsV0(cfg)

    @classmethod
    def from_factor_pack(
        cls,
        pack: RelationalFactorPack,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        redact_law_operator_input: bool = False,
    ) -> RafaRelationalSignatureEncoderV0:
        return cls(
            RafaRelationalSignatureLearningConfig.from_factor_pack(
                pack,
                hidden_dim=hidden_dim,
                dropout=dropout,
                redact_law_operator_input=bool(redact_law_operator_input),
            )
        )

    def encode_input(self, x: torch.Tensor) -> RafaRelationalSignatureHeadOutputs:
        h = _normalized_rows(self.encoder(x.float()))
        return self.heads(h)

    def forward(self, pack: RelationalFactorPack) -> RafaRelationalSignatureHeadOutputs:
        x = factor_pack_to_learning_input(pack, self.cfg)
        return self.encode_input(x)


def matryoshka_signature_losses(
    outputs: RafaRelationalSignatureHeadOutputs,
    targets: RelationalFactorPack,
    weights: RafaSignatureLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or RafaSignatureLossWeights()
    q_target = targets.q_profile.float()
    q_target = q_target / q_target.sum(dim=-1, keepdim=True).clamp_min(EPS)
    law_target = _normalized_rows(targets.law_signature.float())
    op_target = _normalized_rows(targets.operator_seed.float())
    operator_tail = _tail(outputs.h, 640, outputs.h.size(-1))

    q_loss = F.kl_div(outputs.q_profile.clamp_min(EPS).log(), q_target, reduction="batchmean")
    arc_loss = F.mse_loss(outputs.arc_profile, targets.arc_profile.float())
    temporal_loss = F.mse_loss(outputs.temporal_profile, targets.temporal_profile.float())
    support_loss = F.mse_loss(outputs.support_profile, targets.support_profile.float())
    branch_loss = F.mse_loss(outputs.branch_profile, targets.branch_profile.float())
    law_loss = 1.0 - F.cosine_similarity(outputs.law_signature, law_target, dim=-1).mean()
    operator_loss = 1.0 - F.cosine_similarity(outputs.operator_seed, op_target, dim=-1).mean()
    tail_law_loss = law_loss
    tail_operator_loss = operator_loss
    if int(operator_tail.size(-1)) == int(op_target.size(-1)):
        tail_operator_alignment_loss = 1.0 - F.cosine_similarity(operator_tail, op_target, dim=-1).mean()
    else:
        tail_operator_alignment_loss = operator_loss.new_tensor(0.0)
    confidence_loss = F.mse_loss(outputs.confidence, targets.confidence.float())
    unit_norm_loss = (
        (outputs.h.norm(dim=-1) - 1.0).pow(2).mean()
        + (outputs.operator_seed.norm(dim=-1) - 1.0).pow(2).mean()
    )

    total = (
        float(weights.q_profile) * q_loss
        + float(weights.arc_profile) * arc_loss
        + float(weights.temporal_profile) * temporal_loss
        + float(weights.support_profile) * support_loss
        + float(weights.branch_profile) * branch_loss
        + float(weights.law_signature) * law_loss
        + float(weights.operator_seed) * operator_loss
        + float(weights.tail_operator_alignment) * tail_operator_alignment_loss
        + float(weights.confidence) * confidence_loss
        + float(weights.unit_norm) * unit_norm_loss
    )
    return {
        "total_loss": total,
        "q_profile_loss": q_loss,
        "arc_profile_loss": arc_loss,
        "temporal_profile_loss": temporal_loss,
        "support_profile_loss": support_loss,
        "branch_profile_loss": branch_loss,
        "law_signature_loss": law_loss,
        "operator_seed_loss": operator_loss,
        "tail_law_signature_loss": tail_law_loss,
        "tail_operator_seed_loss": tail_operator_loss,
        "tail_operator_alignment_loss": tail_operator_alignment_loss,
        "confidence_loss": confidence_loss,
        "unit_norm_loss": unit_norm_loss,
    }


def summarize_learning_contract(
    outputs: RafaRelationalSignatureHeadOutputs,
    losses: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    loss_summary = {}
    for name, value in (losses or {}).items():
        loss_summary[name] = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    return {
        "h_shape": [int(dim) for dim in outputs.h.shape],
        "q_profile_shape": [int(dim) for dim in outputs.q_profile.shape],
        "arc_profile_shape": [int(dim) for dim in outputs.arc_profile.shape],
        "temporal_profile_shape": [int(dim) for dim in outputs.temporal_profile.shape],
        "support_profile_shape": [int(dim) for dim in outputs.support_profile.shape],
        "branch_profile_shape": [int(dim) for dim in outputs.branch_profile.shape],
        "law_signature_shape": [int(dim) for dim in outputs.law_signature.shape],
        "operator_seed_shape": [int(dim) for dim in outputs.operator_seed.shape],
        "confidence_shape": [int(dim) for dim in outputs.confidence.shape],
        "operator_tail_shape": [int(dim) for dim in _tail(outputs.h, 640, outputs.h.size(-1)).shape],
        "law_operator_readout": "h[640:768]_tail",
        "tail_loss_semantics": "tail_law_signature_loss and tail_operator_seed_loss are aliases of the tail-read head losses; only tail_operator_alignment_loss is an extra direct tail objective.",
        "redact_law_operator_input": bool(getattr(outputs, "redact_law_operator_input", False)),
        "max_h_unit_norm_error": float((outputs.h.norm(dim=-1) - 1.0).abs().max().detach().cpu().item()),
        "max_operator_unit_norm_error": float(
            (outputs.operator_seed.norm(dim=-1) - 1.0).abs().max().detach().cpu().item()
        ),
        "max_law_unit_norm_error": float(
            (outputs.law_signature.norm(dim=-1) - 1.0).abs().max().detach().cpu().item()
        ),
        "max_q_sum_error": float((outputs.q_profile.sum(dim=-1) - 1.0).abs().max().detach().cpu().item()),
        "losses": loss_summary,
    }
