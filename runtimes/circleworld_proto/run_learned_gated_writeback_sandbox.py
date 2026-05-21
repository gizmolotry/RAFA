from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_shadow_branch_law_table import shadow_branch_law_row  # noqa: E402
from circleworld import learned_branch_law_output_names, learned_branch_law_output_tensor  # noqa: E402
from evaluate_shadow_branch_law_calibration import (  # noqa: E402
    _load_model,
    shadow_prediction_to_branch_fields,
)
from train_shadow_branch_law_from_table import shadow_branch_feature_names, shadow_row_to_feature_target  # noqa: E402


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _soft_writeback_scale(score: float, *, soft_floor: float, soft_full: float) -> float:
    if float(soft_full) <= float(soft_floor):
        return 1.0 if float(score) >= float(soft_full) else 0.0
    return _clamp01((float(score) - float(soft_floor)) / (float(soft_full) - float(soft_floor)))


def _candidate_assay_paths(inputs: list[Path]) -> list[Path]:
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
    if not unique:
        raise RuntimeError("No causality assay JSON files found")
    return unique


def learned_gate_row_from_case(
    case: dict[str, Any],
    *,
    predicted_fields: dict[str, float],
    measured_shadow: dict[str, Any],
    writeback_threshold: float,
    gate_mode: str = "hard",
    soft_floor: float = 0.65,
    soft_full: float = 0.88,
    source_path: str = "",
) -> dict[str, Any]:
    writeback_score = _as_float(predicted_fields.get("predicted_writeback_score", 0.0))
    predicted_permission = 1.0 if writeback_score >= writeback_threshold else 0.0
    if str(gate_mode) == "soft":
        writeback_scale = _soft_writeback_scale(writeback_score, soft_floor=float(soft_floor), soft_full=float(soft_full))
    else:
        writeback_scale = predicted_permission
    target_permission = _as_float(measured_shadow.get("writeback_permission", 0.0))
    ungated_parent_div = _as_float(case.get("causal_parent_phase_divergence", 0.0))
    ungated_world_jump = _as_float(case.get("causal_world_jump_proxy", 0.0))
    ungated_support_shift = _as_float(case.get("causal_support_shift", 0.0))
    ungated_qtrace_shift = _as_float(case.get("causal_qtrace_shift_proxy", 0.0))
    gated_parent_div = ungated_parent_div * writeback_scale
    gated_world_jump = ungated_world_jump * writeback_scale
    gated_support_shift = ungated_support_shift * writeback_scale
    gated_qtrace_shift = ungated_qtrace_shift * writeback_scale
    oracle_parent_div = ungated_parent_div if target_permission > 0.0 else 0.0
    oracle_world_jump = ungated_world_jump if target_permission > 0.0 else 0.0
    return {
        "source_path": source_path,
        "source": str(case.get("source", "")),
        "seed": int(_as_float(case.get("seed", 0.0))),
        "status": str(case.get("status", "")),
        "gate_mode": str(gate_mode),
        "target_writeback_permission": target_permission,
        "predicted_writeback_permission": predicted_permission,
        "writeback_score": writeback_score,
        "writeback_scale": writeback_scale,
        "soft_floor": float(soft_floor),
        "soft_full": float(soft_full),
        "writeback_correct": 1.0 if predicted_permission == target_permission else 0.0,
        "writeback_false_permit": 1.0 if predicted_permission > target_permission else 0.0,
        "writeback_false_deny": 1.0 if predicted_permission < target_permission else 0.0,
        "sandbox_label": "learned_permit" if predicted_permission > 0.0 else "learned_deny",
        "target_label": "target_permit" if target_permission > 0.0 else "target_deny",
        "ungated_parent_phase_divergence": ungated_parent_div,
        "ungated_world_jump_proxy": ungated_world_jump,
        "ungated_support_shift": ungated_support_shift,
        "ungated_qtrace_shift_proxy": ungated_qtrace_shift,
        "gated_parent_phase_divergence": gated_parent_div,
        "gated_world_jump_proxy": gated_world_jump,
        "gated_support_shift": gated_support_shift,
        "gated_qtrace_shift_proxy": gated_qtrace_shift,
        "oracle_parent_phase_divergence": oracle_parent_div,
        "oracle_world_jump_proxy": oracle_world_jump,
        "gated_parent_divergence_retained": gated_parent_div / ungated_parent_div if ungated_parent_div > 1.0e-8 else 0.0,
        "gated_world_jump_retained": gated_world_jump / ungated_world_jump if ungated_world_jump > 1.0e-8 else 0.0,
        "world_jump_reduction": ungated_world_jump - gated_world_jump,
        "parent_phase_divergence_reduction": ungated_parent_div - gated_parent_div,
        "oracle_parent_phase_divergence_gap": gated_parent_div - oracle_parent_div,
        "oracle_world_jump_gap": gated_world_jump - oracle_world_jump,
        "predicted_survival_delta": _as_float(predicted_fields.get("predicted_survival_delta", 0.0)),
        "predicted_collapse_pressure": _as_float(predicted_fields.get("predicted_collapse_pressure", 0.0)),
        "predicted_coexistence_drive": _as_float(predicted_fields.get("predicted_coexistence_drive", 0.0)),
        "predicted_q_trace_inheritance_mix": _as_float(predicted_fields.get("predicted_q_trace_inheritance_mix", 0.0)),
        "target_survival_delta": _as_float(measured_shadow.get("survival_delta", 0.0)),
        "target_collapse_pressure": _as_float(measured_shadow.get("collapse_pressure", 0.0)),
    }


def run_learned_gated_writeback_sandbox(
    *,
    checkpoint_path: Path,
    assay_paths: list[Path],
    out_dir: Path,
    device_name: str,
    writeback_threshold: float,
    gate_mode: str,
    soft_floor: float,
    soft_full: float,
) -> dict[str, str]:
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    output_names = list(learned_branch_law_output_names())
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    with torch.no_grad():
        for assay_path in _candidate_assay_paths(assay_paths):
            payload = json.loads(assay_path.read_text(encoding="utf-8"))
            source_rows: list[dict[str, Any]] = []
            for case in payload.get("cases", []) or []:
                measured = shadow_branch_law_row(case, source_path=str(assay_path))
                features, _ = shadow_row_to_feature_target(measured)
                feature_tensor = torch.tensor([features], dtype=torch.float32, device=device)
                pred_tensor = learned_branch_law_output_tensor(model(feature_tensor)).squeeze(0).detach().cpu()
                prediction_by_name = {
                    name: float(pred_tensor[idx].item())
                    for idx, name in enumerate(output_names)
                }
                predicted_fields = shadow_prediction_to_branch_fields(prediction_by_name)
                row = learned_gate_row_from_case(
                    case,
                    predicted_fields=predicted_fields,
                    measured_shadow=measured,
                    writeback_threshold=float(writeback_threshold),
                    gate_mode=str(gate_mode),
                    soft_floor=float(soft_floor),
                    soft_full=float(soft_full),
                    source_path=str(assay_path),
                )
                rows.append(row)
                source_rows.append(row)
            sources.append(_summarize_rows(source_rows, source_path=str(assay_path)))

    aggregate = _summarize_rows(rows, source_path="")
    aggregate.update(
        {
            "status": "pass_learned_gated_writeback_sandbox"
            if rows and aggregate["writeback_accuracy"] >= 0.75
            else "warn_learned_gated_writeback_sandbox",
            "source_count": len(sources),
            "device": str(device),
            "writeback_threshold": float(writeback_threshold),
            "gate_mode": str(gate_mode),
            "soft_floor": float(soft_floor),
            "soft_full": float(soft_full),
        }
    )
    payload = {
        "schema": "learned_gated_writeback_sandbox_v0",
        "checkpoint_path": str(checkpoint_path),
        "feature_names": list(shadow_branch_feature_names()),
        "output_names": output_names,
        "aggregate": aggregate,
        "sources": sources,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "learned_gated_writeback_sandbox.json"
    md_path = out_dir / "LEARNED_GATED_WRITEBACK_SANDBOX.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    return {"json": str(json_path), "markdown": str(md_path), "status": str(aggregate["status"])}


def _summarize_rows(rows: list[dict[str, Any]], *, source_path: str) -> dict[str, Any]:
    ungated_parent = [_as_float(row.get("ungated_parent_phase_divergence", 0.0)) for row in rows]
    gated_parent = [_as_float(row.get("gated_parent_phase_divergence", 0.0)) for row in rows]
    oracle_parent = [_as_float(row.get("oracle_parent_phase_divergence", 0.0)) for row in rows]
    ungated_jump = [_as_float(row.get("ungated_world_jump_proxy", 0.0)) for row in rows]
    gated_jump = [_as_float(row.get("gated_world_jump_proxy", 0.0)) for row in rows]
    oracle_jump = [_as_float(row.get("oracle_world_jump_proxy", 0.0)) for row in rows]
    mean_ungated_parent = _mean(ungated_parent)
    mean_gated_parent = _mean(gated_parent)
    mean_ungated_jump = _mean(ungated_jump)
    mean_gated_jump = _mean(gated_jump)
    return {
        "source_path": source_path,
        "row_count": len(rows),
        "permit_fraction": _mean([_as_float(row.get("predicted_writeback_permission", 0.0)) for row in rows]),
        "target_permit_fraction": _mean([_as_float(row.get("target_writeback_permission", 0.0)) for row in rows]),
        "mean_writeback_scale": _mean([_as_float(row.get("writeback_scale", 0.0)) for row in rows]),
        "writeback_accuracy": _mean([_as_float(row.get("writeback_correct", 0.0)) for row in rows]),
        "false_permit_rate": _mean([_as_float(row.get("writeback_false_permit", 0.0)) for row in rows]),
        "false_deny_rate": _mean([_as_float(row.get("writeback_false_deny", 0.0)) for row in rows]),
        "mean_ungated_parent_phase_divergence": mean_ungated_parent,
        "mean_gated_parent_phase_divergence": mean_gated_parent,
        "mean_oracle_parent_phase_divergence": _mean(oracle_parent),
        "mean_ungated_world_jump_proxy": mean_ungated_jump,
        "mean_gated_world_jump_proxy": mean_gated_jump,
        "mean_oracle_world_jump_proxy": _mean(oracle_jump),
        "parent_divergence_retention_vs_ungated": mean_gated_parent / mean_ungated_parent if mean_ungated_parent > 1.0e-8 else 0.0,
        "world_jump_retention_vs_ungated": mean_gated_jump / mean_ungated_jump if mean_ungated_jump > 1.0e-8 else 0.0,
        "world_jump_reduction_vs_ungated": mean_ungated_jump - mean_gated_jump,
        "parent_divergence_reduction_vs_ungated": mean_ungated_parent - mean_gated_parent,
        "mean_oracle_parent_phase_divergence_gap": _mean(
            [_as_float(row.get("oracle_parent_phase_divergence_gap", 0.0)) for row in rows]
        ),
        "mean_oracle_world_jump_gap": _mean([_as_float(row.get("oracle_world_jump_gap", 0.0)) for row in rows]),
        "mean_writeback_score": _mean([_as_float(row.get("writeback_score", 0.0)) for row in rows]),
        "mean_predicted_survival_delta": _mean([_as_float(row.get("predicted_survival_delta", 0.0)) for row in rows]),
        "mean_predicted_collapse_pressure": _mean(
            [_as_float(row.get("predicted_collapse_pressure", 0.0)) for row in rows]
        ),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Learned-Gated Writeback Sandbox",
        "",
        "This assay applies the learned branch-law checkpoint as a reversible gate over measured causal child writeback.",
        "",
        "## Summary",
        "",
        f"- status: `{agg['status']}`",
        f"- rows: `{agg['row_count']}`",
        f"- sources: `{agg['source_count']}`",
        f"- permit fraction: `{agg['permit_fraction']}`",
        f"- target permit fraction: `{agg['target_permit_fraction']}`",
        f"- mean writeback scale: `{agg['mean_writeback_scale']}`",
        f"- writeback accuracy: `{agg['writeback_accuracy']}`",
        f"- false permit rate: `{agg['false_permit_rate']}`",
        f"- false deny rate: `{agg['false_deny_rate']}`",
        f"- ungated parent divergence: `{agg['mean_ungated_parent_phase_divergence']}`",
        f"- gated parent divergence: `{agg['mean_gated_parent_phase_divergence']}`",
        f"- parent divergence retention: `{agg['parent_divergence_retention_vs_ungated']}`",
        f"- ungated world jump: `{agg['mean_ungated_world_jump_proxy']}`",
        f"- gated world jump: `{agg['mean_gated_world_jump_proxy']}`",
        f"- world jump retention: `{agg['world_jump_retention_vs_ungated']}`",
        f"- world jump reduction: `{agg['world_jump_reduction_vs_ungated']}`",
        "",
        "## Sources",
        "",
        "| source | rows | permit | scale | acc | parent retained | jump retained | ungated jump | gated jump |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in payload.get("sources", []):
        lines.append(
            "| {source} | {rows} | {permit} | {scale} | {acc} | {parent} | {jump} | {uj} | {gj} |".format(
                source=source.get("source_path", ""),
                rows=source.get("row_count", 0),
                permit=source.get("permit_fraction", 0.0),
                scale=source.get("mean_writeback_scale", 0.0),
                acc=source.get("writeback_accuracy", 0.0),
                parent=source.get("parent_divergence_retention_vs_ungated", 0.0),
                jump=source.get("world_jump_retention_vs_ungated", 0.0),
                uj=source.get("mean_ungated_world_jump_proxy", 0.0),
                gj=source.get("mean_gated_world_jump_proxy", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| source | seed | target | pred | score | scale | ungated div | gated div | ungated jump | gated jump |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("rows", []):
        lines.append(
            "| {source} | {seed} | {target} | {pred} | {score} | {scale} | {ud} | {gd} | {uj} | {gj} |".format(
                source=row.get("source_path", ""),
                seed=row.get("seed", 0),
                target=row.get("target_writeback_permission", 0.0),
                pred=row.get("predicted_writeback_permission", 0.0),
                score=row.get("writeback_score", 0.0),
                scale=row.get("writeback_scale", 0.0),
                ud=row.get("ungated_parent_phase_divergence", 0.0),
                gd=row.get("gated_parent_phase_divergence", 0.0),
                uj=row.get("ungated_world_jump_proxy", 0.0),
                gj=row.get("gated_world_jump_proxy", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an assay-only learned gate over measured child writeback outcomes.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--assay", action="append", required=True, help="Causality assay JSON or directory.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--writeback-threshold", type=float, default=0.78)
    ap.add_argument("--gate-mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--soft-floor", type=float, default=0.65)
    ap.add_argument("--soft-full", type=float, default=0.88)
    args = ap.parse_args()
    result = run_learned_gated_writeback_sandbox(
        checkpoint_path=Path(args.checkpoint),
        assay_paths=[Path(raw) for raw in args.assay],
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        writeback_threshold=float(args.writeback_threshold),
        gate_mode=str(args.gate_mode),
        soft_floor=float(args.soft_floor),
        soft_full=float(args.soft_full),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
