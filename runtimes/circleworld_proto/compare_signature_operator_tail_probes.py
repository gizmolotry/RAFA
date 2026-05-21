from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "rafa_signature_operator_tail_probe_compare_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_INPUT_GLOB = str(TOKENBURST_ROOT / "signature_operator_tail_probe*" / "signature_operator_tail_probe.json")
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "signature_operator_tail_probe_compare_2026_05_07"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _best_rows(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rankings = payload.get("rankings") if isinstance(payload.get("rankings"), dict) else {}
    for target, target_rows in rankings.items():
        if not isinstance(target_rows, list) or not target_rows:
            continue
        best = target_rows[0]
        if not isinstance(best, dict):
            continue
        learned_rows = [
            row for row in target_rows if isinstance(row, dict) and str(row.get("view", "")).startswith("learned")
        ]
        explicit = next(
            (row for row in target_rows if isinstance(row, dict) and row.get("view") == "explicit_non_law_metadata"),
            {},
        )
        best_learned = min(learned_rows, key=lambda row: _as_float(row.get("heldout_mse"), float("inf")), default={})
        explicit_mse = _as_float(explicit.get("heldout_mse"), float("inf"))
        best_learned_mse = _as_float(best_learned.get("heldout_mse"), float("inf"))
        rows.append(
            {
                "profile": path.parent.name,
                "path": str(path),
                "seed": payload.get("seed"),
                "case_offset": payload.get("case_offset"),
                "target": target,
                "winning_view": best.get("view"),
                "winning_heldout_mse": best.get("heldout_mse"),
                "winning_relative_improvement": best.get("heldout_relative_improvement"),
                "best_learned_view": best_learned.get("view"),
                "best_learned_heldout_mse": best_learned.get("heldout_mse"),
                "explicit_heldout_mse": explicit.get("heldout_mse"),
                "learned_beats_explicit": best_learned_mse < explicit_mse,
                "learned_minus_explicit_mse": best_learned_mse - explicit_mse,
            }
        )
    return rows


def compare_operator_tail_probes(input_glob: str, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for item in sorted(glob.glob(input_glob)):
        path = Path(item)
        if "smoke" in path.parent.name.lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(_best_rows(payload, path))
        except Exception as exc:  # pragma: no cover - diagnostic utility should report and continue.
            parse_errors.append({"path": str(path), "error": str(exc)})
    learned_wins = [row for row in rows if row.get("learned_beats_explicit")]
    by_target: dict[str, dict[str, Any]] = {}
    for target in sorted({str(row.get("target")) for row in rows}):
        target_rows = [row for row in rows if row.get("target") == target]
        target_wins = [row for row in target_rows if row.get("learned_beats_explicit")]
        by_target[target] = {
            "row_count": len(target_rows),
            "learned_win_count": len(target_wins),
            "learned_win_fraction": len(target_wins) / len(target_rows) if target_rows else 0.0,
            "winning_views": [row.get("winning_view") for row in target_rows],
        }
    status = "operator_tail_probe_missing"
    if rows:
        if len(learned_wins) == len(rows):
            status = "learned_operator_tail_consistently_beats_explicit"
        elif learned_wins:
            status = "operator_tail_mixed_usefulness"
        else:
            status = "explicit_metadata_beats_learned_operator_tail"
    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "status": status,
        "promotion_effect": "none",
        "input_glob": input_glob,
        "row_count": len(rows),
        "learned_win_count": len(learned_wins),
        "learned_win_fraction": len(learned_wins) / len(rows) if rows else 0.0,
        "parse_errors": parse_errors,
        "by_target": by_target,
        "rows": rows,
        "interpretation": (
            "This compares linear predictive usefulness for factor-pack law/operator surfaces only. "
            "It is not an audio continuation proof."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "signature_operator_tail_probe_compare.json"
    md_path = out_dir / "SIGNATURE_OPERATOR_TAIL_PROBE_COMPARE.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else str(value)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Signature Operator-Tail Probe Compare",
        "",
        f"- schema: `{summary['schema']}`",
        f"- status: `{summary['status']}`",
        f"- promotion effect: `{summary['promotion_effect']}`",
        f"- rows: `{summary['row_count']}`",
        f"- learned wins: `{summary['learned_win_count']}`",
        f"- learned win fraction: `{_fmt(summary['learned_win_fraction'])}`",
        "",
        summary["interpretation"],
        "",
        "| profile | seed | offset | target | winner | learned beats explicit | best learned mse | explicit mse | delta mse |",
        "|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {profile} | {seed} | {offset} | {target} | {winner} | {beats} | {learned} | {explicit} | {delta} |".format(
                profile=row.get("profile"),
                seed=_fmt(row.get("seed")),
                offset=_fmt(row.get("case_offset")),
                target=row.get("target"),
                winner=row.get("winning_view"),
                beats=row.get("learned_beats_explicit"),
                learned=_fmt(row.get("best_learned_heldout_mse")),
                explicit=_fmt(row.get("explicit_heldout_mse")),
                delta=_fmt(row.get("learned_minus_explicit_mse")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare signature operator-tail predictive probe outputs.")
    ap.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    summary = compare_operator_tail_probes(str(args.input_glob), Path(args.out_dir))
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "status": summary["status"],
                "saved": str(Path(args.out_dir) / "signature_operator_tail_probe_compare.json"),
                "report": str(Path(args.out_dir) / "SIGNATURE_OPERATOR_TAIL_PROBE_COMPARE.md"),
                "row_count": summary["row_count"],
                "learned_win_count": summary["learned_win_count"],
                "learned_win_fraction": summary["learned_win_fraction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
