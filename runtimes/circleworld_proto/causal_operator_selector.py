from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from resonant_law_objects import LAW_OBJECT_FEATURE_NAMES, law_object_feature_vector

PAIR_SCALAR_NAMES: tuple[str, ...] = (
    "geometric_score",
    "recurrence_compatibility",
    "same_local_family",
    "same_structural_family",
    "identity_membrane_pass",
    "strong_recurrence_pass",
    "query_safety_proxy",
    "candidate_safety_proxy",
)


class CausalOperatorSelector(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).squeeze(-1)


class MultiHeadCausalOperatorSelector(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.route_head = nn.Linear(hidden_dim, 1)
        self.movement_head = nn.Linear(hidden_dim, 1)
        self.jump_safety_head = nn.Linear(hidden_dim, 1)
        self.identity_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(features)
        return {
            "route_score": torch.sigmoid(self.route_head(hidden)).squeeze(-1),
            "movement": torch.sigmoid(self.movement_head(hidden)).squeeze(-1),
            "jump_safety": torch.sigmoid(self.jump_safety_head(hidden)).squeeze(-1),
            "identity": torch.sigmoid(self.identity_head(hidden)).squeeze(-1),
        }


def selector_route_scores(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["route_score"]
    return output


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def causal_pair_feature_vector(query: dict[str, Any], candidate: dict[str, Any], row: dict[str, Any]) -> list[float]:
    q = law_object_feature_vector(query)
    c = law_object_feature_vector(candidate)
    diff = [abs(a - b) for a, b in zip(q, c)]
    prod = [a * b for a, b in zip(q, c)]
    scalars = [_as_float(row.get(name, 0.0)) for name in PAIR_SCALAR_NAMES]
    return list(q) + list(c) + diff + prod + scalars


def causal_pair_feature_dim() -> int:
    return len(LAW_OBJECT_FEATURE_NAMES) * 4 + len(PAIR_SCALAR_NAMES)
