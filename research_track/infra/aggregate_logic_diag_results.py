from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(summary.get("name", "")),
        "checkpoint_name": str(summary.get("checkpoint_name", "")),
        "checkpoint": str(summary.get("checkpoint", "")),
        "anchor_enabled": bool(summary.get("anchor_enabled", False)),
        "intermediate_consistency_enabled": bool(summary.get("intermediate_consistency_enabled", False)),
        "ifs_num_steps": int(summary.get("ifs_num_steps", 0)),
        "num_tasks": int(summary.get("num_tasks", 0)),
        "num_success": int(summary.get("num_success", 0)),
        "success_rate": float(summary.get("success_rate", 0.0)),
        "mean_final_rank": float(summary.get("mean_final_rank", 999.0)),
        "mean_persistence": float(summary.get("mean_persistence", 0.0)),
        "mean_final_margin": float(summary.get("mean_final_margin", 0.0)),
        "mean_energy_gain": float(summary.get("mean_energy_gain", 0.0)),
        "mean_intermediate_consistency_reg": float(summary.get("mean_intermediate_consistency_reg", 0.0)),
        "mean_anchor_lambda": float(summary.get("mean_anchor_lambda", 0.0)),
        "notes": str(summary.get("notes", "")),
        "summary_path": str(summary.get("summary_path", "")),
    }


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -float(row.get("success_rate", 0.0)),
        -float(row.get("mean_persistence", 0.0)),
        float(row.get("mean_final_rank", 999.0)),
        -float(row.get("mean_final_margin", -1e9)),
    )


def build_logic_diag_report(*, repo_root: Path, summaries_dir: Path | None = None, prefix: str = "logic_diag") -> dict[str, Any]:
    summaries_root = summaries_dir or (repo_root / "research_track" / "infra" / "summaries")
    rows: list[dict[str, Any]] = []
    for path in sorted(summaries_root.glob("*.json")):
        if path.name in {"last_logic_diag_batch.json", "logic_diag_leaderboard.json"}:
            continue
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        name = str(summary.get("name", ""))
        if prefix and not name.startswith(prefix):
            continue
        summary["summary_path"] = str(path)
        rows.append(_row_from_summary(summary))

    ranked = sorted(rows, key=_sort_key)
    out = {
        "num_rows": len(ranked),
        "prefix": prefix,
        "rows": ranked,
        "best_by_success_then_persistence": ranked[:10],
    }

    out_json = summaries_root / "logic_diag_leaderboard.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md_lines = [
        "# Logic Diagnostic Leaderboard",
        "",
        f"Rows: {len(ranked)}",
        "",
        "| Name | Ckpt | A | IC | Steps | Success | Mean Persist | Mean Rank | Mean Margin | Mean Gain | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranked:
        md_lines.append(
            "| {name} | {ckpt} | {a} | {ic} | {steps} | {succ} | {persist} | {rank} | {margin} | {gain} | {notes} |".format(
                name=row["name"],
                ckpt=row["checkpoint_name"],
                a=int(bool(row["anchor_enabled"])),
                ic=int(bool(row["intermediate_consistency_enabled"])),
                steps=row["ifs_num_steps"],
                succ=("%.2f" % row["success_rate"]),
                persist=("%.4f" % row["mean_persistence"]),
                rank=("%.2f" % row["mean_final_rank"]),
                margin=("%.4f" % row["mean_final_margin"]),
                gain=("%.4f" % row["mean_energy_gain"]),
                notes=row["notes"].replace("|", "/"),
            )
        )
    out_md = summaries_root / "logic_diag_leaderboard.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    out["json_path"] = str(out_json)
    out["md_path"] = str(out_md)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate logic diagnostic run summaries into a leaderboard.")
    ap.add_argument("--prefix", type=str, default="logic_diag")
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out = build_logic_diag_report(repo_root=repo_root, prefix=args.prefix)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
