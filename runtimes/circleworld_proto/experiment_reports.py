from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


__all__ = [
    "ensure_parent",
    "escape_markdown_table_cell",
    "markdown_table",
    "markdown_table_header",
    "markdown_table_row",
    "report_front_matter",
    "status_bullets",
    "write_markdown",
    "write_text",
]


def ensure_parent(path: str | Path) -> Path:
    """Create a file path's parent directory and return the normalized path."""
    out_path = Path(path)
    parent = out_path.parent
    if parent != Path(""):
        parent.mkdir(parents=True, exist_ok=True)
    return out_path


def write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write plain text after ensuring the destination directory exists."""
    out_path = ensure_parent(path)
    out_path.write_text(text, encoding=encoding)
    return out_path


def write_markdown(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write markdown with a trailing newline for stable report diffs."""
    body = text if text.endswith("\n") else f"{text}\n"
    return write_text(path, body, encoding=encoding)


def escape_markdown_table_cell(value: Any, *, none: str = "") -> str:
    """Format a value for a GitHub-flavored markdown table cell."""
    if value is None:
        text = none
    elif isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def markdown_table_row(cells: Sequence[Any]) -> str:
    """Return one markdown table row from ordered cell values."""
    return "| " + " | ".join(escape_markdown_table_cell(cell) for cell in cells) + " |"


def markdown_table_header(headers: Sequence[str]) -> list[str]:
    """Return header and separator rows for a markdown table."""
    return [
        markdown_table_row(headers),
        "| " + " | ".join("---" for _ in headers) + " |",
    ]


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a complete markdown table."""
    lines = markdown_table_header(headers)
    lines.extend(markdown_table_row(row) for row in rows)
    return "\n".join(lines)


def _front_matter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = "" if value is None else str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def report_front_matter(
    *,
    title: str,
    status: str,
    schema: str | None = None,
    generated_at: datetime | str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Return a small YAML-compatible front matter block for report files."""
    if generated_at is None:
        generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    elif isinstance(generated_at, datetime):
        generated = generated_at.replace(microsecond=0).isoformat()
    else:
        generated = generated_at

    fields: list[tuple[str, Any]] = [
        ("title", title),
        ("status", status),
        ("generated_at", generated),
    ]
    if schema is not None:
        fields.insert(0, ("schema", schema))
    if extra:
        fields.extend(extra.items())

    lines = ["---"]
    lines.extend(f"{key}: {_front_matter_value(value)}" for key, value in fields)
    lines.append("---")
    return "\n".join(lines)


def status_bullets(statuses: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> str:
    """Render report status values as markdown bullets."""
    items = statuses.items() if isinstance(statuses, Mapping) else statuses
    return "\n".join(f"- **{label}:** {escape_markdown_table_cell(value)}" for label, value in items)
