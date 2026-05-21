from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_circleworld import evaluate_config
from export_circleworld_audio import _load_circle_cfg


VARIANTS: dict[str, dict[str, float]] = {
    "baseline_current": {},
    "support_only_phase_operator_floor_off": {
        "child_writeback_phase_delta_gain": 0.0,
        "child_operator_seed_gain": 0.0,
        "child_operator_promotability_gain": 0.0,
        "child_writeback_operator_mix": 0.0,
        "child_writeback_phase_floor_gain": 0.0,
    },
    "raw_phase_only_support_operator_floor_off": {
        "child_support_writeback_gain": 0.0,
        "child_operator_seed_gain": 0.0,
        "child_operator_promotability_gain": 0.0,
        "child_writeback_operator_mix": 0.0,
        "child_writeback_phase_floor_gain": 0.0,
    },
    "operator_floor_only_support_off": {
        "child_support_writeback_gain": 0.0,
        "child_writeback_phase_delta_gain": 0.0,
    },
    "support_operator_no_floor": {
        "child_writeback_phase_floor_gain": 0.0,
    },
    "support_operator_floor_low": {
        "child_writeback_phase_floor_gain": 0.75,
    },
    "support_operator_floor_fixed_2_10": {
        "child_writeback_phase_floor_gain": 2.10,
    },
    "support_operator_floor_high": {
        "child_writeback_phase_floor_gain": 3.25,
    },
}


SUMMARY_KEYS = [
    "mean_major_gain",
    "mean_residue_drop",
    "mean_loss",
    "mean_parent_real_branch_fraction",
    "mean_child_real_branch_fraction",
    "mean_real_branch_fraction",
    "phase_only_real_branch_fraction",
    "phase_only_excess_branch_fraction",
    "mean_child_writeback_mass",
    "mean_child_writeback_gate_mass",
    "mean_child_phase_writeback_delta_mass",
    "mean_child_parent_phase_writeback_delta_mass",
    "mean_child_support_writeback_mass",
    "mean_child_logit_writeback_mass",
    "mean_child_qtrace_writeback_mass",
    "mean_child_parent_divergence",
    "mean_child_sibling_divergence",
    "mean_decorative_slot2_fraction",
]

SOURCE_KEYS = [
    "count",
    "mean_parent_real_branch_fraction",
    "mean_child_real_branch_fraction",
    "mean_real_branch_fraction",
    "phase_only_real_branch_fraction",
    "phase_only_excess_branch_fraction",
    "mean_child_writeback_gate_mass",
    "mean_child_phase_writeback_delta_mass",
    "mean_child_parent_phase_writeback_delta_mass",
    "mean_child_support_writeback_mass",
    "mean_child_logit_writeback_mass",
    "mean_child_qtrace_writeback_mass",
    "mean_child_parent_divergence",
    "final_phase_only_branch_surface_peak",
    "final_phase_only_branch_live_fraction",
    "decorative_slot2_low_phase_fraction",
]


def _jsonable_config(config_path: Path, overrides: dict[str, float]) -> dict[str, Any]:
    cfg = _load_circle_cfg(config_path)
    payload = asdict(cfg)
    payload.update(overrides)
    return {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0",
        "schema_note": "child_writeback_variable_isolation_v2_additive_controls",
        "ablation": "child_writeback_variable_isolation_v2",
        "base_config_path": str(config_path),
        "variant_overrides": overrides,
        "config": payload,
    }


def _select_summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    out = {key: summary.get(key) for key in SUMMARY_KEYS if key in summary}
    by_source = summary.get("by_source", {})
    if isinstance(by_source, dict):
        out["by_source"] = {
            source: {key: row.get(key) for key in SOURCE_KEYS if key in row}
            for source, row in by_source.items()
            if isinstance(row, dict)
        }
    return out


def _write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (int, float)):
            return f"{float(value):.6f}"
        return str(value)

    lines = [
        "# Child Writeback Ablation",
        "",
        "| Variant | Overall phase-only | Parent branch | Child branch | Gate mass | Phase delta | Support writeback | Logit writeback | Major gain | Naked count | Naked phase-only | Naked peak | Naked decorative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row.get("metrics", {})
        naked = metrics.get("by_source", {}).get("naked_rafa")
        naked = naked if isinstance(naked, dict) else {}
        lines.append(
            "| {variant} | {phase} | {parent} | {child} | {gate} | {phase_delta} | {support} | {logit} | {major} | {naked_count} | {naked_phase} | {naked_peak} | {decorative} |".format(
                variant=row["variant"],
                phase=fmt(metrics.get("phase_only_real_branch_fraction")),
                parent=fmt(metrics.get("mean_parent_real_branch_fraction")),
                child=fmt(metrics.get("mean_child_real_branch_fraction")),
                gate=fmt(metrics.get("mean_child_writeback_gate_mass")),
                phase_delta=fmt(metrics.get("mean_child_phase_writeback_delta_mass")),
                support=fmt(metrics.get("mean_child_support_writeback_mass")),
                logit=fmt(metrics.get("mean_child_logit_writeback_mass")),
                major=fmt(metrics.get("mean_major_gain")),
                naked_count=fmt(naked.get("count")),
                naked_phase=fmt(naked.get("phase_only_real_branch_fraction")),
                naked_peak=fmt(naked.get("final_phase_only_branch_surface_peak")),
                decorative=fmt(naked.get("decorative_slot2_low_phase_fraction")),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation contract:",
            "",
            "- `support_only_phase_operator_floor_off` isolates support/logit/q-trace writeback with parent phase writeback disabled.",
            "- `raw_phase_only_support_operator_floor_off` isolates raw child-to-parent phase delta without support/operator/floor writeback.",
            "- `operator_floor_only_support_off` isolates phase operator/floor without support writeback or raw phase-delta writeback.",
            "- `support_operator_no_floor` tests support plus raw/operator phase writeback without the phase-floor threshold push.",
            "- The floor sweep tests whether naked phase branch depends on gain or becomes decorative/overwrite pressure.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(config_path: Path, out_dir: Path, time_steps: int, device: str, write_only: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant, overrides in VARIANTS.items():
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_config_path = variant_dir / "circleworld_config.json"
        variant_payload = _jsonable_config(config_path, overrides)
        variant_config_path.write_text(json.dumps(variant_payload, indent=2), encoding="utf-8")
        row: dict[str, Any] = {
            "variant": variant,
            "config_path": str(variant_config_path),
            "overrides": overrides,
        }
        if not write_only:
            summary = evaluate_config(
                config_path=variant_config_path,
                out_dir=variant_dir,
                time_steps=time_steps,
                device_name=device,
            )
            row["summary_path"] = str(variant_dir / "heldout_summary.json")
            row["metrics"] = _select_summary_metrics(summary)
        rows.append(row)

    report = {
        "base_config_path": str(config_path),
        "time_steps": time_steps,
        "device": device,
        "write_only": write_only,
        "variants": rows,
    }
    (out_dir / "child_writeback_ablation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not write_only:
        _write_markdown(rows, out_dir / "CHILD_WRITEBACK_ABLATION.md")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fixed child-writeback ablations for RAFA variable isolation.")
    ap.add_argument("--config", required=True, help="Base Circleworld config JSON.")
    ap.add_argument("--out-dir", required=True, help="Output directory for variant configs and summaries.")
    ap.add_argument("--time-steps", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write-only", action="store_true", help="Only write variant configs and do not run heldout evaluation.")
    args = ap.parse_args()

    report = run_ablation(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        time_steps=args.time_steps,
        device=args.device,
        write_only=bool(args.write_only),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
