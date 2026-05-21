from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def shadow_branch_law_row(case: dict[str, Any], *, source_path: str = "") -> dict[str, Any]:
    causal_jump = _as_float(case.get("causal_world_jump_proxy", 0.0))
    decoy_jump = _as_float(case.get("decoy_world_jump_proxy", 0.0))
    causal_div = _as_float(case.get("causal_parent_phase_divergence", 0.0))
    decoy_div = _as_float(case.get("decoy_parent_phase_divergence", 0.0))
    route_score = _as_float(case.get("causal_selector_model_score", case.get("causal_selector_hybrid_score", 0.0)))
    identity_signal = max(
        _as_float(case.get("causal_selector_identity_pred", 0.0)),
        _as_float(case.get("causal_selector_identity_membrane_pass", 0.0)),
        _as_float(case.get("causal_selector_same_local_family", 0.0)),
        _as_float(case.get("causal_selector_same_structural_family", 0.0)),
    )
    recurrence = _as_float(case.get("causal_selector_recurrence_compatibility", 0.0))
    jump_safety = _as_float(case.get("causal_selector_jump_safety_pred", 0.0))
    if jump_safety <= 0.0:
        jump_safety = 1.0 / (1.0 + max(0.0, causal_jump))
    support_delta = _as_float(case.get("causal_support_shift", 0.0)) - _as_float(case.get("decoy_support_shift", 0.0))
    qtrace_delta = _as_float(case.get("causal_qtrace_shift_proxy", 0.0)) - _as_float(
        case.get("decoy_qtrace_shift_proxy", 0.0)
    )
    support_stability = _clamp01(1.0 - abs(support_delta))
    qtrace_stability = _clamp01(1.0 - abs(qtrace_delta))
    causal_safety_win = 1.0 if causal_jump < decoy_jump else 0.0
    hard_decoy = 1.0 if decoy_div > causal_div and decoy_jump > causal_jump else 0.0
    movement_ratio = _clamp01(causal_div / (max(causal_div, decoy_div, 1.0e-8)))

    writeback_permission = 1.0 if identity_signal >= 0.5 and route_score >= 0.5 and causal_safety_win > 0.0 else 0.0
    coherence_target = _clamp01(0.45 * identity_signal + 0.35 * recurrence + 0.20 * jump_safety)
    coexistence_drive = _clamp01(0.30 * route_score + 0.25 * identity_signal + 0.25 * recurrence + 0.20 * movement_ratio)
    merge_drive = _clamp01(0.45 * recurrence + 0.35 * identity_signal + 0.20 * support_stability)
    survival_delta = _clamp01(
        0.30 * identity_signal
        + 0.25 * recurrence
        + 0.20 * jump_safety
        + 0.15 * writeback_permission
        + 0.10 * support_stability
    )
    collapse_pressure = _clamp01(1.0 - survival_delta + 0.25 * max(0.0, causal_jump - decoy_jump))
    q_trace_inheritance_mix = _clamp01(0.55 * recurrence + 0.25 * identity_signal + 0.20 * qtrace_stability)

    return {
        "source_path": source_path,
        "source": str(case.get("source", "")),
        "seed": int(_as_float(case.get("seed", 0.0))),
        "status": str(case.get("status", "")),
        "causal_selector_model_score": route_score,
        "identity_signal": identity_signal,
        "recurrence_compatibility": recurrence,
        "jump_safety": jump_safety,
        "coexistence_drive": coexistence_drive,
        "merge_drive": merge_drive,
        "survival_delta": survival_delta,
        "collapse_pressure": collapse_pressure,
        "support_delta": float(support_delta),
        "coherence_target": coherence_target,
        "q_trace_inheritance_mix": q_trace_inheritance_mix,
        "writeback_permission": writeback_permission,
        "hard_decoy_against_causal": hard_decoy,
        "causal_safety_win_over_decoy": causal_safety_win,
        "causal_parent_phase_divergence": causal_div,
        "decoy_parent_phase_divergence": decoy_div,
        "causal_world_jump_proxy": causal_jump,
        "decoy_world_jump_proxy": decoy_jump,
        "decoy_causal_phase_divergence_gap": decoy_div - causal_div,
        "decoy_causal_world_jump_gap": decoy_jump - causal_jump,
    }


def _candidate_json_paths(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in inputs:
        if path.is_dir():
            direct = path / "resonant_operator_causality_assay.json"
            if direct.exists():
                out.append(direct)
            out.extend(sorted(path.glob("**/resonant_operator_causality_assay.json")))
        elif path.exists():
            out.append(path)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in out:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def build_shadow_table(inputs: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for json_path in _candidate_json_paths(inputs):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        aggregate = dict(payload.get("aggregate", {}) or {})
        cases = list(payload.get("cases", []) or [])
        sources.append(
            {
                "path": str(json_path),
                "case_count": len(cases),
                "aggregate": aggregate,
            }
        )
        rows.extend(shadow_branch_law_row(case, source_path=str(json_path)) for case in cases)

    aggregate = {
        "source_count": len(sources),
        "row_count": len(rows),
        "mean_coexistence_drive": _mean([_as_float(row.get("coexistence_drive", 0.0)) for row in rows]),
        "mean_merge_drive": _mean([_as_float(row.get("merge_drive", 0.0)) for row in rows]),
        "mean_survival_delta": _mean([_as_float(row.get("survival_delta", 0.0)) for row in rows]),
        "mean_collapse_pressure": _mean([_as_float(row.get("collapse_pressure", 0.0)) for row in rows]),
        "mean_support_delta": _mean([_as_float(row.get("support_delta", 0.0)) for row in rows]),
        "mean_coherence_target": _mean([_as_float(row.get("coherence_target", 0.0)) for row in rows]),
        "mean_q_trace_inheritance_mix": _mean([_as_float(row.get("q_trace_inheritance_mix", 0.0)) for row in rows]),
        "mean_writeback_permission": _mean([_as_float(row.get("writeback_permission", 0.0)) for row in rows]),
        "mean_hard_decoy_against_causal": _mean([_as_float(row.get("hard_decoy_against_causal", 0.0)) for row in rows]),
        "mean_causal_safety_win_over_decoy": _mean([_as_float(row.get("causal_safety_win_over_decoy", 0.0)) for row in rows]),
        "mean_decoy_causal_phase_divergence_gap": _mean(
            [_as_float(row.get("decoy_causal_phase_divergence_gap", 0.0)) for row in rows]
        ),
        "mean_decoy_causal_world_jump_gap": _mean(
            [_as_float(row.get("decoy_causal_world_jump_gap", 0.0)) for row in rows]
        ),
    }
    return {"aggregate": aggregate, "sources": sources, "rows": rows}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Shadow Branch Law Table",
        "",
        "This table converts causal operator-routing assay outputs into assay-only branch-law fields.",
        "",
        "## Aggregate",
        "",
        f"- sources: `{agg['source_count']}`",
        f"- rows: `{agg['row_count']}`",
        f"- mean coexistence drive: `{agg['mean_coexistence_drive']}`",
        f"- mean merge drive: `{agg['mean_merge_drive']}`",
        f"- mean survival delta: `{agg['mean_survival_delta']}`",
        f"- mean collapse pressure: `{agg['mean_collapse_pressure']}`",
        f"- mean support delta: `{agg['mean_support_delta']}`",
        f"- mean coherence target: `{agg['mean_coherence_target']}`",
        f"- mean q-trace inheritance mix: `{agg['mean_q_trace_inheritance_mix']}`",
        f"- mean writeback permission: `{agg['mean_writeback_permission']}`",
        f"- hard decoy against causal fraction: `{agg['mean_hard_decoy_against_causal']}`",
        f"- causal safety win over decoy fraction: `{agg['mean_causal_safety_win_over_decoy']}`",
        "",
        "## Rows",
        "",
        "| source | seed | survive | collapse | writeback | coexist | merge | q-mix | hard decoy | jump gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {source} | {seed} | {survive} | {collapse} | {writeback} | {coexist} | {merge} | {qmix} | {hard} | {gap} |".format(
                source=row.get("source", ""),
                seed=row.get("seed", 0),
                survive=row.get("survival_delta", 0.0),
                collapse=row.get("collapse_pressure", 0.0),
                writeback=row.get("writeback_permission", 0.0),
                coexist=row.get("coexistence_drive", 0.0),
                merge=row.get("merge_drive", 0.0),
                qmix=row.get("q_trace_inheritance_mix", 0.0),
                hard=row.get("hard_decoy_against_causal", 0.0),
                gap=row.get("decoy_causal_world_jump_gap", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an assay-only shadow branch-law table from causality outputs.")
    ap.add_argument("--input", action="append", required=True, help="JSON file or directory containing assay JSONs.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    payload = build_shadow_table([Path(raw) for raw in args.input])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "shadow_branch_law_table.json"
    md_path = out_dir / "SHADOW_BRANCH_LAW_TABLE.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "aggregate": payload["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
