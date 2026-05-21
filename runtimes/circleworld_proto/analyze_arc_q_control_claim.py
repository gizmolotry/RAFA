from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_INPUT = (
    TOKENBURST_ROOT
    / "claim_isolation_arc_q_extended_smoke_2026_05_07"
    / "claim_isolation_suite_summary.json"
)
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "arc_q_control_claim_2026_05_07"

CORE_METRICS = (
    "mean_parent_real_branch_fraction",
    "mean_child_real_branch_fraction",
    "phase_only_real_branch_fraction",
    "mean_child_support_writeback_mass",
    "mean_child_qtrace_writeback_mass",
    "mean_num_law_families",
    "mean_major_gain",
    "mean_loss",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _top_metrics(row: dict[str, Any]) -> dict[str, float]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    top = metrics.get("top") if isinstance(metrics.get("top"), dict) else {}
    return {key: _as_float(top.get(key)) for key in CORE_METRICS}


def _variant_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tracks = payload.get("tracks") if isinstance(payload.get("tracks"), list) else []
    for track in tracks:
        if isinstance(track, dict) and track.get("track") == "arc_q_cross":
            variants = track.get("variants")
            return [row for row in variants if isinstance(row, dict)] if isinstance(variants, list) else []
    return []


def _drop(base: dict[str, float], other: dict[str, float]) -> dict[str, float]:
    out = {f"{key}_drop": float(base.get(key, 0.0) - other.get(key, 0.0)) for key in CORE_METRICS}
    out["mean_loss_increase"] = float(other.get("mean_loss", 0.0) - base.get("mean_loss", 0.0))
    return out


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    variants = _variant_rows(payload)
    by_q: dict[str, dict[str, dict[str, Any]]] = {}
    for row in variants:
        q_variant = str(row.get("q_variant", "unknown"))
        arc_variant = str(row.get("arc_variant", "unknown"))
        by_q.setdefault(q_variant, {})[arc_variant] = row

    rows: list[dict[str, Any]] = []
    for q_variant, arc_rows in sorted(by_q.items()):
        base = arc_rows.get("arc_baseline")
        if not isinstance(base, dict):
            continue
        base_metrics = _top_metrics(base)
        flat_metrics = _top_metrics(arc_rows.get("arc_flat_major_residue", {}))
        shuffled_metrics = _top_metrics(arc_rows.get("arc_time_shuffled", {}))
        promo_zero_metrics = _top_metrics(arc_rows.get("promotability_zero", {}))
        rows.append(
            {
                "q_variant": q_variant,
                "baseline": base_metrics,
                "arc_flat_major_residue_drop": _drop(base_metrics, flat_metrics),
                "arc_time_shuffled_drop": _drop(base_metrics, shuffled_metrics),
                "promotability_zero_drop": _drop(base_metrics, promo_zero_metrics),
            }
        )

    no_q_relation_variants = {"phase_only_relation_control", "uniform_relation_kernel_control"}
    no_q_relation_rows = [row for row in rows if row["q_variant"] in no_q_relation_variants]
    no_q_relation_arc_gate = all(
        row["baseline"]["mean_parent_real_branch_fraction"] > 0.0
        and row["baseline"]["phase_only_real_branch_fraction"] > 0.0
        and row["arc_flat_major_residue_drop"]["mean_parent_real_branch_fraction_drop"] > 0.0
        and row["promotability_zero_drop"]["mean_parent_real_branch_fraction_drop"] > 0.0
        for row in no_q_relation_rows
    )
    ramanujan = next((row for row in rows if row["q_variant"] == "ramanujan_baseline"), None)
    qtrace = next((row for row in rows if row["q_variant"] == "qtrace_only_control"), None)
    phase_only = next((row for row in rows if row["q_variant"] == "phase_only_relation_control"), None)
    relation_parent_spread = 0.0
    if ramanujan and qtrace and phase_only:
        vals = [
            ramanujan["baseline"]["mean_parent_real_branch_fraction"],
            qtrace["baseline"]["mean_parent_real_branch_fraction"],
            phase_only["baseline"]["mean_parent_real_branch_fraction"],
        ]
        relation_parent_spread = float(max(vals) - min(vals))
    status = (
        "arc_gateway_supported_without_q_relation"
        if no_q_relation_arc_gate and relation_parent_spread <= 0.005
        else "arc_q_claim_needs_review"
    )
    return {
        "schema": "rafa_arc_q_control_claim_v0",
        "status": status,
        "source_summary": str(payload.get("out_dir", "")),
        "time_steps": payload.get("time_steps"),
        "seed_plan": payload.get("seed_plan"),
        "q_variant_count": len(rows),
        "no_q_relation_arc_gate": bool(no_q_relation_arc_gate),
        "ramanujan_qtrace_phase_parent_spread": relation_parent_spread,
        "rows": rows,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Arc/Q Control Claim",
        "",
        f"- schema: `{payload['schema']}`",
        f"- status: `{payload['status']}`",
        f"- q variants: `{payload['q_variant_count']}`",
        f"- no-q-relation arc gate: `{payload['no_q_relation_arc_gate']}`",
        f"- Ramanujan/qtrace/phase parent spread: `{payload['ramanujan_qtrace_phase_parent_spread']:.6f}`",
        "",
        "| q variant | base parent | base child | base phase | flat parent drop | promo-zero parent drop | shuffled parent drop | loss inc flat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("rows", []):
        base = row["baseline"]
        flat = row["arc_flat_major_residue_drop"]
        promo = row["promotability_zero_drop"]
        shuffled = row["arc_time_shuffled_drop"]
        lines.append(
            "| {q} | {parent} | {child} | {phase} | {flat_parent} | {promo_parent} | {shuf_parent} | {loss_inc} |".format(
                q=row["q_variant"],
                parent=_fmt(base["mean_parent_real_branch_fraction"]),
                child=_fmt(base["mean_child_real_branch_fraction"]),
                phase=_fmt(base["phase_only_real_branch_fraction"]),
                flat_parent=_fmt(flat["mean_parent_real_branch_fraction_drop"]),
                promo_parent=_fmt(promo["mean_parent_real_branch_fraction_drop"]),
                shuf_parent=_fmt(shuffled["mean_parent_real_branch_fraction_drop"]),
                loss_inc=_fmt(flat["mean_loss_increase"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze whether arc/promotability causality survives q/relation controls.")
    ap.add_argument("--summary-json", default=str(DEFAULT_INPUT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    payload = analyze(_load_json(Path(args.summary_json)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "arc_q_control_claim.json"
    md_path = out_dir / "ARC_Q_CONTROL_CLAIM.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
