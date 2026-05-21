from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


def ensure_parent(path: Path) -> None:
    """Create a file path's parent directory when it exists."""
    parent = Path(path).parent
    if parent != Path(""):
        parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, *, encoding: str = "utf-8-sig", require_object: bool = True) -> dict[str, Any] | Any:
    payload = json.loads(Path(path).read_text(encoding=encoding))
    if require_object and not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    encoding: str = "utf-8",
) -> None:
    ensure_parent(Path(path))
    Path(path).write_text(json.dumps(payload, indent=indent, sort_keys=sort_keys), encoding=encoding)


def write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    ensure_parent(Path(path))
    Path(path).write_text(text, encoding=encoding)


def as_float(value: Any, default: float = 0.0, *, nonfinite_default: bool = True) -> float:
    """Coerce metrics to float without importing optional tensor libraries."""
    try:
        if value is None:
            return float(default)
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
            cpu = getattr(value, "cpu", None)
            if callable(cpu):
                value = cpu()
            if getattr(value, "ndim", 0) != 0:
                mean_fn = getattr(value, "mean", None)
                if callable(mean_fn):
                    value = mean_fn()
            item = getattr(value, "item", None)
            if callable(item):
                value = item()
        out = float(value)
    except (TypeError, ValueError, RuntimeError):
        return float(default)
    if nonfinite_default and not math.isfinite(out):
        return float(default)
    return out


def mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else float(default)


def median(values: Iterable[float], default: float = 0.0) -> float:
    vals = sorted(float(value) for value in values)
    if not vals:
        return float(default)
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def win_fraction(values: Iterable[float], *, threshold: float = 0.0, default: float = 0.0) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return float(default)
    return float(sum(1 for value in vals if value > threshold) / len(vals))


def fmt(value: Any, *, none: str = "NA", format_ints: bool = True, precision: int = 6) -> str:
    if value is None:
        return none
    if isinstance(value, (int, float)) if format_ints else isinstance(value, float):
        return f"{float(value):.{precision}g}"
    return str(value)


def fmt_float_only(value: Any, *, precision: int = 6) -> str:
    return fmt(value, none="None", format_ints=False, precision=precision)


def parse_csv_floats(raw: str | None, default: Sequence[float] = ()) -> list[float]:
    if raw is None or not str(raw).strip():
        return [float(value) for value in default]
    values: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def parse_required_csv_floats(raw: str | None, *, empty_message: str) -> list[float]:
    values = parse_csv_floats(raw)
    if not values:
        raise ValueError(empty_message)
    return values


def parse_gain_csv(
    raw: str | None,
    *,
    empty_message: str = "At least one gain is required",
    include_zero: bool = True,
    zero_epsilon: float | None = 1.0e-12,
) -> list[float]:
    values = parse_required_csv_floats(raw, empty_message=empty_message)
    if include_zero:
        has_zero = any(abs(value) < zero_epsilon for value in values) if zero_epsilon is not None else 0.0 in values
        if not has_zero:
            values.insert(0, 0.0)
    return values


def safe_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return str(requested)
    try:
        import torch  # type: ignore

        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        return "cpu"
