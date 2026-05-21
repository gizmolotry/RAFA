from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Mapping, Sequence


DEFAULT_SOURCE_BASE_SEEDS: dict[str, int] = {
    "synthetic": 5100,
    "naked_rafa": 6100,
}

BASE_SEED_PRESETS: dict[str, dict[str, int]] = {
    "heldout": DEFAULT_SOURCE_BASE_SEEDS,
    "train": {
        "synthetic": 1100,
        "naked_rafa": 3100,
    },
    "validation": {
        "synthetic": 2100,
        "naked_rafa": 4100,
    },
    "real_anchor_probe": {
        "synthetic": 7100,
        "naked_rafa": 8100,
    },
    "seeded_branch_active": {
        "naked_rafa": 9100,
    },
    "branch_recovery_soft_guard_v1": {
        "naked_rafa": 9200,
    },
}

SOURCE_ALIASES: dict[str, str] = {
    "synth": "synthetic",
    "synthetic": "synthetic",
    "synthetic_phase": "synthetic",
    "naked": "naked_rafa",
    "naked-rafa": "naked_rafa",
    "naked_rafa": "naked_rafa",
    "rafa": "naked_rafa",
}

DEFAULT_SOURCE_ORDER: tuple[str, ...] = ("synthetic", "naked_rafa")

_CASE_KEY_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_INT_RANGE_RE = re.compile(r"^\s*([+-]?\d+)\s*(?:\.\.|-|:)\s*([+-]?\d+)\s*$")


@dataclass(frozen=True)
class ParsedSeedSpec:
    kind: str
    values: tuple[int, ...] = ()
    count: int | None = None

    @property
    def is_explicit(self) -> bool:
        return self.kind == "explicit"

    @property
    def is_count(self) -> bool:
        return self.kind == "count"


@dataclass(frozen=True)
class SeedPlanRecord:
    source: str
    seed: int
    case_key: str
    ordinal: int
    source_ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[str, int]:
        return (self.source, int(self.seed))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_seed_source(source: str) -> str:
    raw = str(source or "").strip()
    key = raw.lower().replace(" ", "_")
    normalized = SOURCE_ALIASES.get(key, key)
    if not normalized:
        raise ValueError("seed source must be non-empty")
    return normalized


def parse_seed_sources(seed_source: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(seed_source, str):
        normalized = seed_source.strip().lower()
        if normalized == "both":
            return DEFAULT_SOURCE_ORDER
        parts = [part.strip() for part in re.split(r"[,;\s]+", seed_source) if part.strip()]
    else:
        parts = [str(part) for part in seed_source]
    sources = tuple(normalize_seed_source(part) for part in parts)
    if not sources:
        raise ValueError("at least one seed source is required")
    return sources


def normalize_case_key(value: Any, *, fallback: str = "case") -> str:
    text = str(value if value is not None else "").strip().lower()
    text = _CASE_KEY_CLEAN_RE.sub("_", text).strip("_")
    if text:
        return text
    return _CASE_KEY_CLEAN_RE.sub("_", str(fallback).strip().lower()).strip("_") or "case"


def seed_case_key(source: str, seed: int, *, prefix: str | None = None) -> str:
    parts = [prefix, normalize_seed_source(source), f"seed_{int(seed)}"]
    return normalize_case_key("_".join(str(part) for part in parts if part))


def parse_seed_count(value: Any, *, option_name: str = "seed_count") -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{option_name} must be an integer, got {value!r}") from exc
    if count < 0:
        raise ValueError(f"{option_name} must be non-negative, got {count}")
    return count


def parse_explicit_seed_list(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(_parse_seed_item(item) for item in value)

    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("["):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON seed list: {value!r}") from exc
        if not isinstance(raw, list):
            raise ValueError(f"JSON seed list must decode to a list, got {type(raw).__name__}")
        return parse_explicit_seed_list(raw)

    seeds: list[int] = []
    for token in [part for part in re.split(r"[,;\s]+", text) if part]:
        range_match = _INT_RANGE_RE.match(token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(_parse_seed_item(token))
    return tuple(seeds)


def parse_seed_list_or_count(value: Any, *, single_value_is_count: bool = True) -> ParsedSeedSpec:
    if value is None:
        raise ValueError("seed spec is required")
    if isinstance(value, int):
        return ParsedSeedSpec(kind="count", count=parse_seed_count(value)) if single_value_is_count else ParsedSeedSpec(
            kind="explicit",
            values=(int(value),),
        )
    if isinstance(value, Sequence) and not isinstance(value, str):
        return ParsedSeedSpec(kind="explicit", values=parse_explicit_seed_list(value))

    text = str(value).strip()
    lowered = text.lower()
    for prefix in ("count:", "count=", "n:", "n="):
        if lowered.startswith(prefix):
            return ParsedSeedSpec(kind="count", count=parse_seed_count(text[len(prefix) :].strip()))
    for prefix in ("seed:", "seed=", "seeds:", "seeds=", "list:", "list="):
        if lowered.startswith(prefix):
            return ParsedSeedSpec(kind="explicit", values=parse_explicit_seed_list(text[len(prefix) :].strip()))

    explicit_markers = (",", ";", " ", "[", "..")
    looks_like_range = bool(_INT_RANGE_RE.match(text))
    if any(marker in text for marker in explicit_markers) or looks_like_range:
        return ParsedSeedSpec(kind="explicit", values=parse_explicit_seed_list(text))
    if single_value_is_count:
        return ParsedSeedSpec(kind="count", count=parse_seed_count(text))
    return ParsedSeedSpec(kind="explicit", values=parse_explicit_seed_list(text))


def source_base_seed(
    source: str,
    *,
    base_seeds: Mapping[str, int] | None = None,
    preset: str | None = None,
) -> int:
    normalized = normalize_seed_source(source)
    resolved = _resolve_base_seeds(base_seeds=base_seeds, preset=preset)
    if normalized not in resolved:
        known = ", ".join(sorted(resolved))
        raise KeyError(f"no base seed configured for source {normalized!r}; known sources: {known}")
    return int(resolved[normalized])


def source_seed_range(
    source: str,
    count: int,
    *,
    base_seeds: Mapping[str, int] | None = None,
    preset: str | None = None,
) -> tuple[int, ...]:
    resolved_count = parse_seed_count(count)
    start = source_base_seed(source, base_seeds=base_seeds, preset=preset)
    return tuple(start + idx for idx in range(resolved_count))


def resolve_seed_values(
    source: str,
    *,
    seeds: Any = None,
    seed_count: Any = None,
    base_seeds: Mapping[str, int] | None = None,
    preset: str | None = None,
    single_value_is_count: bool = True,
) -> tuple[int, ...]:
    if seeds is not None and seed_count is not None:
        raise ValueError("provide either explicit seeds or seed_count, not both")
    if seed_count is not None:
        return source_seed_range(source, parse_seed_count(seed_count), base_seeds=base_seeds, preset=preset)
    if seeds is None:
        raise ValueError("provide explicit seeds or seed_count")

    parsed = parse_seed_list_or_count(seeds, single_value_is_count=single_value_is_count)
    if parsed.is_count:
        return source_seed_range(source, int(parsed.count or 0), base_seeds=base_seeds, preset=preset)
    return tuple(int(seed) for seed in parsed.values)


def build_seed_plan(
    *,
    seed_source: str | Sequence[str],
    seeds: Any = None,
    seed_count: Any = None,
    source_seeds: Mapping[str, Any] | None = None,
    source_seed_counts: Mapping[str, Any] | None = None,
    base_seeds: Mapping[str, int] | None = None,
    preset: str | None = None,
    case_key_prefix: str | None = None,
    single_value_is_count: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> list[SeedPlanRecord]:
    sources = parse_seed_sources(seed_source)
    records: list[SeedPlanRecord] = []
    base_metadata = dict(metadata or {})
    for source in sources:
        source_seeds_value = _lookup_source_value(source_seeds, source)
        source_count_value = _lookup_source_value(source_seed_counts, source)
        values = resolve_seed_values(
            source,
            seeds=source_seeds_value if source_seeds_value is not None else seeds,
            seed_count=source_count_value if source_count_value is not None else seed_count,
            base_seeds=base_seeds,
            preset=preset,
            single_value_is_count=single_value_is_count,
        )
        for source_ordinal, seed in enumerate(values):
            records.append(
                SeedPlanRecord(
                    source=source,
                    seed=int(seed),
                    case_key=seed_case_key(source, int(seed), prefix=case_key_prefix),
                    ordinal=len(records),
                    source_ordinal=source_ordinal,
                    metadata=dict(base_metadata),
                )
            )
    return records


def normalize_seed_plan(raw_plan: Sequence[Any], *, case_key_prefix: str | None = None) -> list[SeedPlanRecord]:
    records: list[SeedPlanRecord] = []
    by_source_counts: dict[str, int] = {}
    for item in raw_plan:
        if isinstance(item, SeedPlanRecord):
            source = item.source
            seed = int(item.seed)
            metadata = dict(item.metadata)
            case_key = normalize_case_key(item.case_key or seed_case_key(source, seed, prefix=case_key_prefix))
        elif isinstance(item, Mapping):
            source = normalize_seed_source(str(item.get("source") or item.get("seed_source") or ""))
            seed = _parse_seed_item(item.get("seed"))
            metadata = {key: value for key, value in item.items() if key not in {"source", "seed_source", "seed", "case_key"}}
            case_key = normalize_case_key(item.get("case_key") or seed_case_key(source, seed, prefix=case_key_prefix))
        else:
            source_raw, seed_raw = item
            source = normalize_seed_source(str(source_raw))
            seed = _parse_seed_item(seed_raw)
            metadata = {}
            case_key = seed_case_key(source, seed, prefix=case_key_prefix)
        source_ordinal = by_source_counts.get(source, 0)
        by_source_counts[source] = source_ordinal + 1
        records.append(
            SeedPlanRecord(
                source=source,
                seed=seed,
                case_key=case_key,
                ordinal=len(records),
                source_ordinal=source_ordinal,
                metadata=metadata,
            )
        )
    return records


def seed_plan_tuples(records: Sequence[SeedPlanRecord]) -> list[tuple[str, int]]:
    return [record.as_tuple() for record in records]


def seed_plan_dicts(records: Sequence[SeedPlanRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


def seed_plan_summary(records: Sequence[SeedPlanRecord]) -> dict[str, Any]:
    by_source: dict[str, list[int]] = {}
    for record in records:
        by_source.setdefault(record.source, []).append(int(record.seed))
    return {
        "count": len(records),
        "sources": {
            source: {
                "count": len(values),
                "min_seed": min(values) if values else None,
                "max_seed": max(values) if values else None,
                "seeds": list(values),
            }
            for source, values in sorted(by_source.items())
        },
        "case_keys": [record.case_key for record in records],
    }


def _parse_seed_item(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed must be an integer, got {value!r}") from exc


def _resolve_base_seeds(
    *,
    base_seeds: Mapping[str, int] | None = None,
    preset: str | None = None,
) -> dict[str, int]:
    if base_seeds is not None:
        return {normalize_seed_source(source): int(seed) for source, seed in base_seeds.items()}
    if preset is None:
        return dict(DEFAULT_SOURCE_BASE_SEEDS)
    key = normalize_case_key(preset)
    if key not in BASE_SEED_PRESETS:
        known = ", ".join(sorted(BASE_SEED_PRESETS))
        raise KeyError(f"unknown base seed preset {preset!r}; known presets: {known}")
    return {normalize_seed_source(source): int(seed) for source, seed in BASE_SEED_PRESETS[key].items()}


def _lookup_source_value(mapping: Mapping[str, Any] | None, source: str) -> Any:
    if mapping is None:
        return None
    if source in mapping:
        return mapping[source]
    for key, value in mapping.items():
        if normalize_seed_source(str(key)) == source:
            return value
    return None


__all__ = [
    "BASE_SEED_PRESETS",
    "DEFAULT_SOURCE_BASE_SEEDS",
    "DEFAULT_SOURCE_ORDER",
    "ParsedSeedSpec",
    "SeedPlanRecord",
    "build_seed_plan",
    "normalize_case_key",
    "normalize_seed_plan",
    "normalize_seed_source",
    "parse_explicit_seed_list",
    "parse_seed_count",
    "parse_seed_list_or_count",
    "parse_seed_sources",
    "resolve_seed_values",
    "seed_case_key",
    "seed_plan_dicts",
    "seed_plan_summary",
    "seed_plan_tuples",
    "source_base_seed",
    "source_seed_range",
]
