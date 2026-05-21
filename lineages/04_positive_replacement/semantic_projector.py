from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SemanticProjectorConfig:
    embedding_dim: int = 768
    control_dim: int = 5
    control_names: tuple[str, ...] = (
        "harmonic_coupling",
        "decay_rate",
        "noise_injection",
        "branch_temperature",
        "support_spread",
    )
    output_gain: float = 1.0


@dataclass
class SemanticControlProfile:
    harmonic_coupling: float
    decay_rate: float
    noise_injection: float
    branch_temperature: float
    support_spread: float
    confidence: float
    control_vector: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "harmonic_coupling": float(self.harmonic_coupling),
            "decay_rate": float(self.decay_rate),
            "noise_injection": float(self.noise_injection),
            "branch_temperature": float(self.branch_temperature),
            "support_spread": float(self.support_spread),
            "confidence": float(self.confidence),
        }

    def to_boundary_conditions(self) -> dict[str, float]:
        return {
            "harmonic_coupling": float(self.harmonic_coupling),
            "decay_rate": float(self.decay_rate),
            "noise_injection": float(self.noise_injection),
            "branch_temperature": float(self.branch_temperature),
            "support_spread": float(self.support_spread),
        }


@dataclass
class SemanticProjectorFactors:
    coarse: torch.Tensor | None = None
    q: torch.Tensor | None = None
    temporal: torch.Tensor | None = None
    support: torch.Tensor | None = None
    branch: torch.Tensor | None = None
    operator_seed: torch.Tensor | None = None

    def named_factors(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, value in (
            ("coarse", self.coarse),
            ("q", self.q),
            ("temporal", self.temporal),
            ("support", self.support),
            ("branch", self.branch),
            ("operator_seed", self.operator_seed),
            ("operator", self.operator_seed),
        ):
            if value is not None:
                out[name] = value
        return out


ALLOWED_PROMPT_FAMILIES: tuple[str, ...] = (
    "airplane",
    "reed",
    "bell",
    "impact",
)

PROJECTOR_FACTOR_NAMES: tuple[str, ...] = (
    "coarse",
    "q",
    "temporal",
    "support",
    "branch",
    "operator_seed",
)


def verify_five_control_contract(cfg: SemanticProjectorConfig | None = None) -> SemanticProjectorConfig:
    cfg = cfg or SemanticProjectorConfig()
    if int(cfg.control_dim) != 5:
        raise ValueError(f"semantic projector control_dim must be 5, got {int(cfg.control_dim)}")
    if len(tuple(cfg.control_names)) != 5:
        raise ValueError(f"semantic projector requires exactly 5 control names, got {len(tuple(cfg.control_names))}")
    return cfg


PROMPT_FAMILY_TEXT: dict[str, str] = {
    "airplane": "broad engine surge, sustained thrust, noisy aerodynamic wash",
    "reed": "stable reed excitation, sustained tone, moderate noise, narrow support",
    "bell": "bright struck resonance, high harmonic coupling, long decay tail",
    "impact": "sharp transient hit, fast decay, wide initial support, low sustain",
}


def projector_contract(cfg: SemanticProjectorConfig | None = None) -> dict[str, Any]:
    cfg = verify_five_control_contract(cfg)
    return {
        "embedding_dim": int(cfg.embedding_dim),
        "control_dim": int(cfg.control_dim),
        "control_names": list(cfg.control_names),
        "factor_names": list(PROJECTOR_FACTOR_NAMES),
        "allowed_prompt_families": list(ALLOWED_PROMPT_FAMILIES),
        "prompt_family_text": dict(PROMPT_FAMILY_TEXT),
    }


def prompt_family_text(prompt_family: str) -> str:
    key = str(prompt_family).strip().lower()
    if key not in PROMPT_FAMILY_TEXT:
        raise KeyError(f"unsupported prompt family: {prompt_family}")
    return PROMPT_FAMILY_TEXT[key]


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    if vector.numel() == 0:
        return vector
    return torch.nn.functional.normalize(vector.view(1, -1), dim=-1).view(-1)


def _stretch_vector(vector: torch.Tensor, out_dim: int) -> torch.Tensor:
    if vector.numel() == out_dim:
        return vector
    return torch.nn.functional.interpolate(
        vector.view(1, 1, -1),
        size=int(out_dim),
        mode="linear",
        align_corners=True,
    ).view(-1)


def _flatten_factor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"factor '{name}' must be a torch.Tensor")
    if value.dim() == 0:
        return value.view(1)
    return value.reshape(-1)


def _canonical_factor_name(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "coarse_profile": "coarse",
        "q_profile": "q",
        "temporal_profile": "temporal",
        "support_profile": "support",
        "branch_profile": "branch",
        "operator": "operator_seed",
    }
    return aliases.get(key, key)


def _coerce_projector_factors(
    factors: SemanticProjectorFactors | dict[str, torch.Tensor] | None = None,
    *,
    operator_seed: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    pool: dict[str, torch.Tensor] = {}
    if factors is not None:
        source = factors.named_factors() if isinstance(factors, SemanticProjectorFactors) else dict(factors)
        for raw_name, value in source.items():
            if value is None:
                continue
            name = _canonical_factor_name(raw_name)
            if name in PROJECTOR_FACTOR_NAMES:
                pool[name] = value
    if operator_seed is not None:
        pool["operator_seed"] = operator_seed
    return pool


def build_projector_embedding(
    embedding: torch.Tensor | None = None,
    *,
    factors: SemanticProjectorFactors | dict[str, torch.Tensor] | None = None,
    factor_names: tuple[str, ...] | list[str] | None = None,
    operator_seed: torch.Tensor | None = None,
    cfg: SemanticProjectorConfig | None = None,
) -> torch.Tensor:
    cfg = cfg or SemanticProjectorConfig()
    if embedding is not None:
        if embedding.dim() != 1:
            raise ValueError("embedding must have shape [D]")
        if embedding.numel() != cfg.embedding_dim:
            raise ValueError(f"embedding dim mismatch: expected {cfg.embedding_dim}, got {embedding.numel()}")
        return embedding

    pool = _coerce_projector_factors(factors, operator_seed=operator_seed)
    if not pool:
        raise ValueError("projector embedding requires either embedding, operator_seed, or factor tensors")

    if factor_names is None:
        selected_names = ("operator_seed",) if "operator_seed" in pool else tuple(pool.keys())
    else:
        selected_names = tuple(_canonical_factor_name(name) for name in factor_names)
    missing = [name for name in selected_names if name not in pool]
    if missing:
        raise KeyError(f"missing projector factors: {missing}")

    source = torch.cat([_flatten_factor(name, pool[name]) for name in selected_names], dim=0)
    source = _normalize_vector(source)
    return _normalize_vector(_stretch_vector(source, cfg.embedding_dim))


def build_factor_projector_embedding(
    *,
    factors: SemanticProjectorFactors | dict[str, torch.Tensor] | None = None,
    factor_names: tuple[str, ...] | list[str] | None = None,
    operator_seed: torch.Tensor | None = None,
    cfg: SemanticProjectorConfig | None = None,
) -> torch.Tensor:
    return build_projector_embedding(
        None,
        factors=factors,
        factor_names=factor_names,
        operator_seed=operator_seed,
        cfg=cfg,
    )


def project_embedding_to_controls(
    embedding: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    cfg: SemanticProjectorConfig | None = None,
) -> SemanticControlProfile:
    cfg = verify_five_control_contract(cfg)
    if embedding.dim() != 1:
        raise ValueError("embedding must have shape [D]")
    if embedding.numel() != cfg.embedding_dim:
        raise ValueError(f"embedding dim mismatch: expected {cfg.embedding_dim}, got {embedding.numel()}")
    if weight.shape != (cfg.control_dim, cfg.embedding_dim):
        raise ValueError(
            f"weight shape mismatch: expected {(cfg.control_dim, cfg.embedding_dim)}, got {tuple(weight.shape)}"
        )
    projected = weight.to(embedding.device, embedding.dtype) @ embedding
    if bias is not None:
        if bias.shape != (cfg.control_dim,):
            raise ValueError(f"bias shape mismatch: expected {(cfg.control_dim,)}, got {tuple(bias.shape)}")
        projected = projected + bias.to(embedding.device, embedding.dtype)
    control = torch.sigmoid(cfg.output_gain * projected)
    confidence = float(control.mean().item())
    return SemanticControlProfile(
        harmonic_coupling=float(control[0].item()),
        decay_rate=float(control[1].item()),
        noise_injection=float(control[2].item()),
        branch_temperature=float(control[3].item()),
        support_spread=float(control[4].item()),
        confidence=confidence,
        control_vector=control.detach(),
    )


def project_factors_to_controls(
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    cfg: SemanticProjectorConfig | None = None,
    embedding: torch.Tensor | None = None,
    factors: SemanticProjectorFactors | dict[str, torch.Tensor] | None = None,
    factor_names: tuple[str, ...] | list[str] | None = None,
    operator_seed: torch.Tensor | None = None,
) -> SemanticControlProfile:
    cfg = verify_five_control_contract(cfg)
    projector_embedding = (
        build_projector_embedding(
            embedding,
            factors=factors,
            factor_names=factor_names,
            operator_seed=operator_seed,
            cfg=cfg,
        )
        if embedding is not None
        else build_factor_projector_embedding(
            factors=factors,
            factor_names=factor_names,
            operator_seed=operator_seed,
            cfg=cfg,
        )
    )
    return project_embedding_to_controls(projector_embedding, weight=weight, bias=bias, cfg=cfg)


def semantic_profile_from_state_dict(
    embedding: torch.Tensor | None,
    projector_state: dict[str, Any],
    cfg: SemanticProjectorConfig | None = None,
    *,
    factors: SemanticProjectorFactors | dict[str, torch.Tensor] | None = None,
    factor_names: tuple[str, ...] | list[str] | None = None,
    operator_seed: torch.Tensor | None = None,
) -> SemanticControlProfile:
    cfg = verify_five_control_contract(cfg)
    if "weight" not in projector_state:
        raise KeyError("projector_state must contain 'weight'")
    weight = projector_state["weight"]
    bias = projector_state.get("bias")
    return project_factors_to_controls(
        weight=weight,
        bias=bias,
        cfg=cfg,
        embedding=embedding,
        factors=factors,
        factor_names=factor_names,
        operator_seed=operator_seed,
    )
