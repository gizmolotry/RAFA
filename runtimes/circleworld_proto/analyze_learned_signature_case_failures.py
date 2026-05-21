from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "rafa_learned_signature_case_failure_atlas_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_INPUT_GLOB = str(TOKENBURST_ROOT / "learned_signature_scout*" / "learned_signature_scout.json")
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "learned_signature_case_failures_2026_05_07"
STABILITY_EPSILON = 1.0e-4
SEPARATION_EPSILON = 1.0e-4


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _case_delta(case: dict[str, Any], family: str) -> float:
    block = _dict(case.get(family))
    return _as_float(block.get("delta_learned_minus_metadata"))


def _row(path: Path, case: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = _dict(payload.get("verdict"))
    stability_delta = _case_delta(case, "stability")
    separation_delta = _case_delta(case, "separation")
    return {
        "profile": path.parent.name,
        "path": str(path),
        "case": str(case.get("case") or "unknown"),
        "split_mode": payload.get("split_mode"),
        "seed": payload.get("seed"),
        "case_offset": payload.get("case_offset"),
        "status": verdict.get("status"),
        "stability_delta": stability_delta,
        "separation_delta": separation_delta,
        "stability_pass": stability_delta > STABILITY_EPSILON,
        "separation_pass": separation_delta > SEPARATION_EPSILON,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_case(case: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    stability = [_as_float(row.get("stability_delta")) for row in rows]
    separation = [_as_float(row.get("separation_delta")) for row in rows]
    worst = min(rows, key=lambda row: _as_float(row.get("stability_delta")), default={})
    best = max(rows, key=lambda row: _as_float(row.get("stability_delta")), default={})
    return {
        "case": case,
        "row_count": len(rows),
        "stability_pass_count": sum(1 for row in rows if row.get("stability_pass")),
        "stability_pass_fraction": (
            sum(1 for row in rows if row.get("stability_pass")) / len(rows) if rows else 0.0
        ),
        "mean_stability_delta": _mean(stability),
        "worst_stability_delta": min(stability) if stability else 0.0,
        "best_stability_delta": max(stability) if stability else 0.0,
        "mean_separation_delta": _mean(separation),
        "worst_profile": worst.get("profile"),
        "best_profile": best.get("profile"),
        "rows": sorted(rows, key=lambda row: _as_float(row.get("stability_delta"))),
    }


def analyze_case_failures(input_glob: str, out_dir: Path) -> dict[str, Any]:
    paths = [Path(item) for item in sorted(glob.glob(input_glob))]
    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not str(payload.get("split_mode", "")).startswith("heldout"):
                continue
            for case in payload.get("heldout_cases", []):
                if isinstance(case, dict):
                    rows.append(_row(path, case))
        except Exception as exc:  # pragma: no cover - diagnostic utility should report and continue.
            parse_errors.append({"path": str(path), "error": str(exc)})
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row.get("case")), []).append(row)
    case_rows = [_summarize_case(case, case_rows) for case, case_rows in sorted(by_case.items())]
    case_rows.sort(key=lambda row: (row["mean_stability_delta"], row["stability_pass_fraction"]))
    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "input_glob": input_glob,
        "heldout_case_row_count": len(rows),
        "case_count": len(case_rows),
        "parse_errors": parse_errors,
        "status": "case_failure_atlas_built" if rows else "no_heldout_case_rows",
        "worst_cases": case_rows[:10],
        "case_rows": case_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "learned_signature_case_failures.json"
    md_path = out_dir / "LEARNED_SIGNATURE_CASE_FAILURES.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.6f}"
    return "" if value is None else str(value)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Learned Signature Case Failure Atlas",
        "",
        f"- schema: `{summary['schema']}`",
        f"- status: `{summary['status']}`",
        f"- heldout case rows: `{summary['heldout_case_row_count']}`",
        f"- cases: `{summary['case_count']}`",
        "",
        "| case | rows | stability pass | mean stability d | worst stability d | best stability d | mean separation d | worst profile |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("case_rows", []):
        lines.append(
            "| {case} | {rows} | {pass_count}/{rows} | {mean_stable} | {worst} | {best} | {mean_sep} | {worst_profile} |".format(
                case=row.get("case"),
                rows=row.get("row_count"),
                pass_count=row.get("stability_pass_count"),
                mean_stable=_fmt(row.get("mean_stability_delta")),
                worst=_fmt(row.get("worst_stability_delta")),
                best=_fmt(row.get("best_stability_delta")),
                mean_sep=_fmt(row.get("mean_separation_delta")),
                worst_profile=row.get("worst_profile"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a per-case heldout failure atlas for learned signature scouts.")
    ap.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    summary = analyze_case_failures(str(args.input_glob), Path(args.out_dir))
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "status": summary["status"],
                "saved": str(Path(args.out_dir) / "learned_signature_case_failures.json"),
                "report": str(Path(args.out_dir) / "LEARNED_SIGNATURE_CASE_FAILURES.md"),
                "heldout_case_row_count": summary["heldout_case_row_count"],
                "case_count": summary["case_count"],
                "worst_cases": [row.get("case") for row in summary.get("worst_cases", [])[:5]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
