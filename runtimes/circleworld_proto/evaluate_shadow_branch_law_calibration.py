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
from circleworld import (  # noqa: E402
    RafaLearnedBranchLawV0,
    learned_branch_law_output_names,
    learned_branch_law_output_tensor,
)
from train_shadow_branch_law_from_table import (  # noqa: E402
    shadow_branch_feature_names,
    shadow_row_to_feature_target,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _threshold_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = [round(0.50 + 0.01 * idx, 2) for idx in range(46)]
    scored: list[dict[str, float]] = []
    for threshold in thresholds:
        correct = 0
        false_permit = 0
        false_deny = 0
        for row in rows:
            pred = 1.0 if _as_float(row.get("writeback_score", 0.0)) >= threshold else 0.0
            target = _as_float(row.get("target_writeback_permission", 0.0))
            correct += 1 if pred == target else 0
            false_permit += 1 if pred > target else 0
            false_deny += 1 if pred < target else 0
        denom = max(1, len(rows))
        scored.append(
            {
                "threshold": float(threshold),
                "accuracy": float(correct / denom),
                "false_permit_rate": float(false_permit / denom),
                "false_deny_rate": float(false_deny / denom),
            }
        )
    best = max(
        scored,
        key=lambda row: (
            row["accuracy"],
            -row["false_permit_rate"],
            -abs(row["threshold"] - 0.78),
        ),
    )
    return {"best": best, "grid": scored}


def shadow_prediction_to_branch_fields(prediction_by_name: dict[str, float]) -> dict[str, float]:
    survival_signed = 0.5 * (
        _as_float(prediction_by_name.get("survival_delta0", 0.0))
        + _as_float(prediction_by_name.get("survival_delta1", 0.0))
    )
    survival_delta = _clamp01(0.5 * (survival_signed + 1.0))
    collapse_pressure = _clamp01(_as_float(prediction_by_name.get("collapse_pressure", 0.0)))
    coexistence_drive = _clamp01(_as_float(prediction_by_name.get("coexistence_drive", 0.0)))
    merge_drive = _clamp01(_as_float(prediction_by_name.get("merge_drive", 0.0)))
    qtrace_mix = _clamp01(_as_float(prediction_by_name.get("qtrace_inheritance_mix", 0.0)))
    coherence_target = _clamp01(_as_float(prediction_by_name.get("coherence_target", 0.0)))
    support_delta = 0.5 * (
        _as_float(prediction_by_name.get("support_delta0", 0.0))
        + _as_float(prediction_by_name.get("support_delta1", 0.0))
    )
    writeback_score = _clamp01(0.50 * survival_delta + 0.30 * (1.0 - collapse_pressure) + 0.20 * coexistence_drive)
    return {
        "predicted_survival_delta": survival_delta,
        "predicted_collapse_pressure": collapse_pressure,
        "predicted_support_delta": float(max(-1.0, min(1.0, support_delta))),
        "predicted_coexistence_drive": coexistence_drive,
        "predicted_merge_drive": merge_drive,
        "predicted_q_trace_inheritance_mix": qtrace_mix,
        "predicted_coherence_target": coherence_target,
        "predicted_writeback_score": writeback_score,
    }


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


def _load_model(checkpoint_path: Path, device: torch.device) -> RafaLearnedBranchLawV0:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_names = list(checkpoint.get("feature_names", []))
    output_names = list(checkpoint.get("output_names", []))
    expected_features = list(shadow_branch_feature_names())
    expected_outputs = list(learned_branch_law_output_names())
    if feature_names and feature_names != expected_features:
        raise RuntimeError(f"Feature name mismatch: checkpoint={feature_names} expected={expected_features}")
    if output_names and output_names != expected_outputs:
        raise RuntimeError(f"Output name mismatch: checkpoint={output_names} expected={expected_outputs}")
    model = RafaLearnedBranchLawV0(
        input_dim=int(checkpoint.get("input_dim", len(expected_features))),
        hidden_dim=int(checkpoint.get("hidden_dim", 64)),
        output_dim=int(checkpoint.get("output_dim", len(expected_outputs))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_shadow_branch_law_calibration(
    *,
    checkpoint_path: Path,
    assay_paths: list[Path],
    out_dir: Path,
    device_name: str,
    writeback_threshold: float,
) -> dict[str, str]:
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    output_names = list(learned_branch_law_output_names())
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    with torch.no_grad():
        for assay_path in _candidate_assay_paths(assay_paths):
            payload = json.loads(assay_path.read_text(encoding="utf-8"))
            cases = list(payload.get("cases", []) or [])
            source_rows: list[dict[str, Any]] = []
            for case in cases:
                measured = shadow_branch_law_row(case, source_path=str(assay_path))
                features, targets = shadow_row_to_feature_target(measured)
                feature_tensor = torch.tensor([features], dtype=torch.float32, device=device)
                pred_tensor = learned_branch_law_output_tensor(model(feature_tensor)).squeeze(0).detach().cpu()
                prediction_by_name = {
                    name: float(pred_tensor[idx].item())
                    for idx, name in enumerate(output_names)
                }
                branch_fields = shadow_prediction_to_branch_fields(prediction_by_name)
                predicted_writeback = 1.0 if branch_fields["predicted_writeback_score"] >= float(writeback_threshold) else 0.0
                target_by_name = {name: float(targets[idx]) for idx, name in enumerate(output_names)}
                output_abs_errors = {
                    name: abs(float(prediction_by_name[name]) - float(target_by_name[name]))
                    for name in output_names
                }
                row = {
                    "source_path": str(assay_path),
                    "source": str(measured.get("source", "")),
                    "seed": int(_as_float(measured.get("seed", 0.0))),
                    "target_writeback_permission": _as_float(measured.get("writeback_permission", 0.0)),
                    "predicted_writeback_permission": predicted_writeback,
                    "writeback_score": branch_fields["predicted_writeback_score"],
                    "writeback_correct": 1.0 if predicted_writeback == _as_float(measured.get("writeback_permission", 0.0)) else 0.0,
                    "writeback_false_permit": 1.0
                    if predicted_writeback > _as_float(measured.get("writeback_permission", 0.0))
                    else 0.0,
                    "writeback_false_deny": 1.0
                    if predicted_writeback < _as_float(measured.get("writeback_permission", 0.0))
                    else 0.0,
                    "writeback_brier": float(
                        (branch_fields["predicted_writeback_score"] - _as_float(measured.get("writeback_permission", 0.0)))
                        ** 2
                    ),
                    "target_survival_delta": _as_float(measured.get("survival_delta", 0.0)),
                    "target_collapse_pressure": _as_float(measured.get("collapse_pressure", 0.0)),
                    "target_support_delta": _as_float(measured.get("support_delta", 0.0)),
                    "target_coexistence_drive": _as_float(measured.get("coexistence_drive", 0.0)),
                    "target_q_trace_inheritance_mix": _as_float(measured.get("q_trace_inheritance_mix", 0.0)),
                    **branch_fields,
                    "survival_delta_abs_error": abs(
                        branch_fields["predicted_survival_delta"] - _as_float(measured.get("survival_delta", 0.0))
                    ),
                    "collapse_pressure_abs_error": abs(
                        branch_fields["predicted_collapse_pressure"] - _as_float(measured.get("collapse_pressure", 0.0))
                    ),
                    "support_delta_abs_error": abs(
                        branch_fields["predicted_support_delta"] - _as_float(measured.get("support_delta", 0.0))
                    ),
                    "coexistence_drive_abs_error": abs(
                        branch_fields["predicted_coexistence_drive"] - _as_float(measured.get("coexistence_drive", 0.0))
                    ),
                    "q_trace_inheritance_mix_abs_error": abs(
                        branch_fields["predicted_q_trace_inheritance_mix"]
                        - _as_float(measured.get("q_trace_inheritance_mix", 0.0))
                    ),
                    "mean_output_abs_error": _mean(list(output_abs_errors.values())),
                    "max_output_abs_error": max(output_abs_errors.values()) if output_abs_errors else 0.0,
                    "output_abs_errors": output_abs_errors,
                    "prediction_by_name": prediction_by_name,
                    "target_by_name": target_by_name,
                }
                rows.append(row)
                source_rows.append(row)
            sources.append(
                {
                    "path": str(assay_path),
                    "row_count": len(source_rows),
                    "writeback_accuracy": _mean([_as_float(row.get("writeback_correct", 0.0)) for row in source_rows]),
                    "mean_output_abs_error": _mean([_as_float(row.get("mean_output_abs_error", 0.0)) for row in source_rows]),
                    "mean_survival_delta_abs_error": _mean(
                        [_as_float(row.get("survival_delta_abs_error", 0.0)) for row in source_rows]
                    ),
                    "mean_collapse_pressure_abs_error": _mean(
                        [_as_float(row.get("collapse_pressure_abs_error", 0.0)) for row in source_rows]
                    ),
                }
            )

    threshold_sweep = _threshold_sweep(rows)
    positive_scores = [
        _as_float(row.get("writeback_score", 0.0))
        for row in rows
        if _as_float(row.get("target_writeback_permission", 0.0)) > 0.0
    ]
    negative_scores = [
        _as_float(row.get("writeback_score", 0.0))
        for row in rows
        if _as_float(row.get("target_writeback_permission", 0.0)) <= 0.0
    ]
    aggregate = {
        "status": "pass_shadow_branch_law_calibration"
        if rows and _mean([_as_float(row.get("writeback_correct", 0.0)) for row in rows]) >= 0.75
        else "warn_shadow_branch_law_calibration",
        "row_count": len(rows),
        "source_count": len(sources),
        "device": str(device),
        "writeback_threshold": float(writeback_threshold),
        "writeback_accuracy": _mean([_as_float(row.get("writeback_correct", 0.0)) for row in rows]),
        "writeback_false_permit_rate": _mean([_as_float(row.get("writeback_false_permit", 0.0)) for row in rows]),
        "writeback_false_deny_rate": _mean([_as_float(row.get("writeback_false_deny", 0.0)) for row in rows]),
        "mean_writeback_brier": _mean([_as_float(row.get("writeback_brier", 0.0)) for row in rows]),
        "mean_positive_writeback_score": _mean(positive_scores),
        "mean_negative_writeback_score": _mean(negative_scores),
        "writeback_score_margin": _mean(positive_scores) - _mean(negative_scores) if positive_scores and negative_scores else 0.0,
        "best_writeback_threshold": threshold_sweep["best"]["threshold"],
        "best_writeback_threshold_accuracy": threshold_sweep["best"]["accuracy"],
        "best_writeback_threshold_false_permit_rate": threshold_sweep["best"]["false_permit_rate"],
        "best_writeback_threshold_false_deny_rate": threshold_sweep["best"]["false_deny_rate"],
        "mean_output_abs_error": _mean([_as_float(row.get("mean_output_abs_error", 0.0)) for row in rows]),
        "max_output_abs_error": max([_as_float(row.get("max_output_abs_error", 0.0)) for row in rows] + [0.0]),
        "mean_survival_delta_abs_error": _mean([_as_float(row.get("survival_delta_abs_error", 0.0)) for row in rows]),
        "mean_collapse_pressure_abs_error": _mean(
            [_as_float(row.get("collapse_pressure_abs_error", 0.0)) for row in rows]
        ),
        "mean_support_delta_abs_error": _mean([_as_float(row.get("support_delta_abs_error", 0.0)) for row in rows]),
        "mean_coexistence_drive_abs_error": _mean(
            [_as_float(row.get("coexistence_drive_abs_error", 0.0)) for row in rows]
        ),
        "mean_q_trace_inheritance_mix_abs_error": _mean(
            [_as_float(row.get("q_trace_inheritance_mix_abs_error", 0.0)) for row in rows]
        ),
    }
    payload = {
        "schema": "shadow_branch_law_calibration_v0",
        "checkpoint_path": str(checkpoint_path),
        "feature_names": list(shadow_branch_feature_names()),
        "output_names": output_names,
        "aggregate": aggregate,
        "threshold_sweep": threshold_sweep,
        "sources": sources,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "shadow_branch_law_calibration.json"
    md_path = out_dir / "SHADOW_BRANCH_LAW_CALIBRATION.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    return {"json": str(json_path), "markdown": str(md_path), "status": aggregate["status"]}


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Shadow Branch Law Calibration",
        "",
        "This evaluates a learned branch-law checkpoint against raw causality assay cases before table aggregation.",
        "",
        "## Summary",
        "",
        f"- status: `{agg['status']}`",
        f"- rows: `{agg['row_count']}`",
        f"- sources: `{agg['source_count']}`",
        f"- writeback threshold: `{agg['writeback_threshold']}`",
        f"- writeback accuracy: `{agg['writeback_accuracy']}`",
        f"- false permit rate: `{agg['writeback_false_permit_rate']}`",
        f"- false deny rate: `{agg['writeback_false_deny_rate']}`",
        f"- mean writeback brier: `{agg['mean_writeback_brier']}`",
        f"- mean positive writeback score: `{agg['mean_positive_writeback_score']}`",
        f"- mean negative writeback score: `{agg['mean_negative_writeback_score']}`",
        f"- writeback score margin: `{agg['writeback_score_margin']}`",
        f"- best threshold: `{agg['best_writeback_threshold']}`",
        f"- best threshold accuracy: `{agg['best_writeback_threshold_accuracy']}`",
        f"- mean output abs error: `{agg['mean_output_abs_error']}`",
        f"- max output abs error: `{agg['max_output_abs_error']}`",
        f"- mean survival error: `{agg['mean_survival_delta_abs_error']}`",
        f"- mean collapse error: `{agg['mean_collapse_pressure_abs_error']}`",
        "",
        "## Sources",
        "",
        "| source | rows | writeback acc | output MAE | survival MAE | collapse MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in payload.get("sources", []):
        lines.append(
            "| {source} | {rows} | {acc} | {mae} | {surv} | {collapse} |".format(
                source=source.get("path", ""),
                rows=source.get("row_count", 0),
                acc=source.get("writeback_accuracy", 0.0),
                mae=source.get("mean_output_abs_error", 0.0),
                surv=source.get("mean_survival_delta_abs_error", 0.0),
                collapse=source.get("mean_collapse_pressure_abs_error", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| source | seed | target wb | pred wb | wb score | survival err | collapse err | output MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("rows", []):
        lines.append(
            "| {source} | {seed} | {target} | {pred} | {score} | {surv} | {collapse} | {mae} |".format(
                source=row.get("source_path", ""),
                seed=row.get("seed", 0),
                target=row.get("target_writeback_permission", 0.0),
                pred=row.get("predicted_writeback_permission", 0.0),
                score=row.get("writeback_score", 0.0),
                surv=row.get("survival_delta_abs_error", 0.0),
                collapse=row.get("collapse_pressure_abs_error", 0.0),
                mae=row.get("mean_output_abs_error", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate a shadow learned branch-law checkpoint on raw causality assays.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--assay", action="append", required=True, help="Causality assay JSON or directory.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--writeback-threshold", type=float, default=0.78)
    args = ap.parse_args()
    result = evaluate_shadow_branch_law_calibration(
        checkpoint_path=Path(args.checkpoint),
        assay_paths=[Path(raw) for raw in args.assay],
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        writeback_threshold=float(args.writeback_threshold),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
