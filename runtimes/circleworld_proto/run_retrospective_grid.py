from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_circleworld_real_anchor import _load_cases_json, run_benchmark
from evaluate_circleworld import evaluate_config


DEFAULT_EXPANDED_CASES = ROOT / "outputs" / "circleworld_proto" / "expanded_anchor_cases_2026-04-09.json"


def _config_id(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    return f"{parent}__{stem}".replace(".", "_")


def _discover_configs(root: Path) -> list[Path]:
    configs = sorted(root.rglob("*.json"))
    keep: list[Path] = []
    for path in configs:
        name = path.name
        if not (name.startswith("circleworld_") or name.startswith("circleworld_real_anchor_")):
            continue
        keep.append(path)
    return keep


def _maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _train_summary_for_config(config_path: Path) -> dict[str, Any] | None:
    run_name = config_path.parent.name
    summary_path = ROOT / "outputs" / "circleworld_proto" / run_name / "train_summary.json"
    return _maybe_load_json(summary_path)


def _baseline_row(train_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not train_summary:
        return {}
    out: dict[str, Any] = {}
    for key in ("baseline_val", "best_val", "baseline_train", "best_train"):
        block = train_summary.get(key)
        if isinstance(block, dict):
            for metric in (
                "mean_score",
                "mean_loss",
                "mean_corr",
                "mean_mae",
                "mean_mse",
                "mean_major_gain",
                "mean_residue_drop",
                "mean_promotability_gain",
                "mean_dominant_q_share",
                "mean_q_entropy",
            ):
                if metric in block:
                    out[f"{key}__{metric}"] = block[metric]
    return out


def _real_audio_score(block: dict[str, Any]) -> float:
    return float(block["mean_corr"] - 0.5 * block["mean_mae"] - 0.25 * block["mean_mse"])


def _structural_score(block: dict[str, Any]) -> float:
    return float(
        block["mean_major_gain"]
        + block["mean_residue_drop"]
        + 0.5 * block["mean_promotability_gain"]
        - 0.1 * block["mean_l_q_dom"]
        - 0.1 * block["mean_l_major_sat"]
    )


def _rank_desc(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda r: float(r[key]), reverse=True)
    return {row["config_id"]: idx + 1 for idx, row in enumerate(ordered)}


def _rank_asc(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda r: float(r[key]))
    return {row["config_id"]: idx + 1 for idx, row in enumerate(ordered)}


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    top_audio = sorted(rows, key=lambda r: r["expanded_real_audio_score"], reverse=True)[:5]
    top_struct = sorted(rows, key=lambda r: r["heldout_structural_score"], reverse=True)[:5]
    current = next((r for r in rows if "realanchor_constrained" in r["config_id"]), None)

    lines = [
        "# Circleworld Retrospective Grid",
        "",
        f"- configs evaluated: {len(rows)}",
        "",
        "## Top Real-Audio Runs (Expanded Benchmark)",
        "",
    ]
    for row in top_audio:
        lines.append(
            f"- `{row['config_id']}`: corr={row['expanded_mean_corr']:.4f}, mae={row['expanded_mean_mae']:.4f}, "
            f"mse={row['expanded_mean_mse']:.5f}, score={row['expanded_real_audio_score']:.4f}"
        )

    lines.extend(["", "## Top Structural Runs (Held-Out Phase Eval)", ""])
    for row in top_struct:
        lines.append(
            f"- `{row['config_id']}`: structural={row['heldout_structural_score']:.4f}, "
            f"major={row['heldout_mean_major_gain']:.4f}, residue={row['heldout_mean_residue_drop']:.4f}, "
            f"entropy={row['heldout_mean_q_entropy']:.4f}"
        )

    lines.extend(["", "## Current Baseline", ""])
    if current is not None:
        lines.append(
            f"- `{current['config_id']}`: expanded audio rank={current['rank_expanded_audio']}, "
            f"heldout structural rank={current['rank_heldout_structural']}, "
            f"expanded corr={current['expanded_mean_corr']:.4f}, heldout structural={current['heldout_structural_score']:.4f}"
        )
    else:
        lines.append("- current constrained baseline not found in this grid")

    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Real-audio score here is a simple composite over expanded benchmark correlation, MAE, and MSE.",
            "- Structural score here is a simple composite over held-out major gain, residue drop, promotability gain, and penalty terms.",
            "- This report is for retrospective comparison only; it does not replace listening.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grid(
    config_root: Path,
    out_dir: Path,
    device_name: str,
    clip_seconds: int,
    phase_blend: float,
    time_steps: int,
    expanded_cases_path: Path,
) -> dict[str, Any]:
    configs = _discover_configs(config_root)
    expanded_cases = _load_cases_json(expanded_cases_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for config_path in configs:
        config_id = _config_id(config_path)
        run_dir = out_dir / config_id
        standard_summary_path = run_dir / "benchmark_standard" / "benchmark_summary.json"
        expanded_summary_path = run_dir / "benchmark_expanded" / "benchmark_summary.json"
        heldout_summary_path = run_dir / "heldout_eval" / "heldout_summary.json"

        standard = _maybe_load_json(standard_summary_path)
        if standard is None:
            standard = run_benchmark(
                config_path=config_path,
                out_dir=run_dir / "benchmark_standard",
                device_name=device_name,
                clip_seconds=clip_seconds,
                phase_blend=phase_blend,
                rerender_check=True,
            )

        expanded = _maybe_load_json(expanded_summary_path)
        if expanded is None:
            expanded = run_benchmark(
                config_path=config_path,
                out_dir=run_dir / "benchmark_expanded",
                device_name=device_name,
                clip_seconds=clip_seconds,
                phase_blend=phase_blend,
                rerender_check=True,
                cases=expanded_cases,
            )

        heldout = _maybe_load_json(heldout_summary_path)
        if heldout is None:
            heldout = evaluate_config(
                config_path=config_path,
                out_dir=run_dir / "heldout_eval",
                time_steps=time_steps,
                device_name=device_name,
            )
        train_summary = _train_summary_for_config(config_path)
        row = {
            "config_id": config_id,
            "config_path": str(config_path),
            "run_family": config_path.parent.name,
            "schema_hint": "real_anchor" if "real_anchor" in config_path.name else "mixed_seed",
            "standard_mean_corr": standard["mean_corr"],
            "standard_mean_mae": standard["mean_mae"],
            "standard_mean_mse": standard["mean_mse"],
            "expanded_mean_corr": expanded["mean_corr"],
            "expanded_mean_mae": expanded["mean_mae"],
            "expanded_mean_mse": expanded["mean_mse"],
            "expanded_bitwise_stable": expanded["all_circleworld_bitwise_stable"],
            "heldout_mean_major_gain": heldout["mean_major_gain"],
            "heldout_mean_residue_drop": heldout["mean_residue_drop"],
            "heldout_mean_promotability_gain": heldout["mean_promotability_gain"],
            "heldout_mean_dominant_q_share": heldout["mean_dominant_q_share"],
            "heldout_mean_q_entropy": heldout["mean_q_entropy"],
            "heldout_mean_l_q_dom": heldout["mean_l_q_dom"],
            "heldout_mean_l_major_sat": heldout["mean_l_major_sat"],
            "standard_real_audio_score": _real_audio_score(standard),
            "expanded_real_audio_score": _real_audio_score(expanded),
            "heldout_structural_score": _structural_score(heldout),
            **_baseline_row(train_summary),
        }
        rows.append(row)

    rank_expanded = _rank_desc(rows, "expanded_real_audio_score")
    rank_struct = _rank_desc(rows, "heldout_structural_score")
    rank_corr = _rank_desc(rows, "expanded_mean_corr")
    rank_mae = _rank_asc(rows, "expanded_mean_mae")

    for row in rows:
        row["rank_expanded_audio"] = rank_expanded[row["config_id"]]
        row["rank_heldout_structural"] = rank_struct[row["config_id"]]
        row["rank_expanded_corr"] = rank_corr[row["config_id"]]
        row["rank_expanded_mae"] = rank_mae[row["config_id"]]

    rows_sorted = sorted(rows, key=lambda r: (r["rank_expanded_audio"], r["rank_heldout_structural"], r["config_id"]))

    summary = {
        "runtime": "circleworld_proto",
        "mode": "retrospective_grid",
        "config_count": len(rows_sorted),
        "device": device_name,
        "clip_seconds": clip_seconds,
        "phase_blend": phase_blend,
        "time_steps": time_steps,
        "expanded_cases_path": str(expanded_cases_path),
        "rows": rows_sorted,
    }

    json_path = out_dir / "retrospective_grid_summary.json"
    csv_path = out_dir / "retrospective_grid_summary.csv"
    md_path = out_dir / "RETROSPECTIVE_GRID.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows_sorted for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    _write_markdown(md_path, rows_sorted)

    return {
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "summary_md": str(md_path),
        "config_count": len(rows_sorted),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full retrospective Circleworld comparison grid.")
    ap.add_argument("--config-root", default=str(ROOT / "checkpoints_circleworld_proto"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--time-steps", type=int, default=128)
    ap.add_argument("--expanded-cases-json", default=str(DEFAULT_EXPANDED_CASES))
    args = ap.parse_args()

    result = run_grid(
        config_root=Path(args.config_root),
        out_dir=Path(args.out_dir),
        device_name=args.device,
        clip_seconds=args.clip_seconds,
        phase_blend=args.phase_blend,
        time_steps=args.time_steps,
        expanded_cases_path=Path(args.expanded_cases_json),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
