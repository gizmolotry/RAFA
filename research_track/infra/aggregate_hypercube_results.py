from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    scorecard = dict(summary.get("path_b_scorecard") or {})
    return {
        "name": str(summary.get("name", "")),
        "model_type": str(summary.get("model_type", "")),
        "memory_enabled": bool(summary.get("memory_enabled", True)),
        "ablate_ramanujan": bool(summary.get("ablate_ramanujan", False)),
        "ablate_slow_clock": bool(summary.get("ablate_slow_clock", False)),
        "hyena_conductor_enabled": bool(summary.get("hyena_conductor_enabled", True)),
        "metamer_enabled": bool(summary.get("metamer_enabled", True)),
        "text_prompt_conditioned": bool(summary.get("text_prompt_conditioned", False)),
        "text_condition_mode": str(summary.get("text_condition_mode", "")),
        "semantic_condition_mode": str(summary.get("semantic_condition_mode", "none")),
        "checkpoint": str(summary.get("checkpoint", "")),
        "collapse_rate_pct": scorecard.get("collapse_rate_pct"),
        "avg_top_ratio": scorecard.get("avg_top_ratio"),
        "avg_cent_var": scorecard.get("avg_cent_var"),
        "avg_dphi_std": scorecard.get("avg_dphi_std"),
        "avg_audio_effect": scorecard.get("avg_audio_effect"),
        "avg_state_effect": scorecard.get("avg_state_effect"),
        "avg_coupling_score": scorecard.get("avg_coupling_score"),
        "notes": str(summary.get("notes", "")),
        "summary_path": str(summary.get("summary_path", "")),
    }


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    collapse = float(row.get("collapse_rate_pct") if row.get("collapse_rate_pct") is not None else 1e9)
    coupling = -float(row.get("avg_coupling_score") if row.get("avg_coupling_score") is not None else -1e9)
    top_ratio = -float(row.get("avg_top_ratio") if row.get("avg_top_ratio") is not None else -1e9)
    return (collapse, coupling, top_ratio)


def build_hypercube_report(
    *,
    repo_root: Path,
    summaries_dir: Path | None = None,
    prefix: str = "pathb_hypercube",
) -> dict[str, Any]:
    summaries_root = summaries_dir or (repo_root / "research_track" / "infra" / "summaries")
    extra_named_runs = {
        "baseline_v3_anchor",
        "prompt_conditioned_clap_smoke",
        "prompt_conditioned_ifs_control_smoke",
        "prompt_conditioned_ifs_control_smoke_to120",
        "prompt_conditioned_hybrid_smoke",
        "semantic_tension_steering_smoke",
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(summaries_root.glob("*.json")):
        if path.name in {"last_batch_run.json", "last_airflow_summary.json", "hypercube_leaderboard.json"}:
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(summary.get("path_b_scorecard"), dict):
            continue
        name = str(summary.get("name", ""))
        if prefix and not (name.startswith(prefix) or name in extra_named_runs):
            continue
        summary["summary_path"] = str(path)
        rows.append(_row_from_summary(summary))

    ranked = sorted(rows, key=_sort_key)
    out = {
        "num_rows": len(ranked),
        "prefix": prefix,
        "rows": ranked,
        "best_by_collapse_then_coupling": ranked[:10],
    }

    out_json = summaries_root / "hypercube_leaderboard.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md_lines = [
        "# Hypercube Leaderboard",
        "",
        f"Rows: {len(ranked)}",
        "",
        "| Name | Collapse % | Coupling | AudioEff | StateEff | TopRatio | CentVar | TextCond | Semantic | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranked:
        md_lines.append(
            "| {name} | {collapse} | {coupling} | {audio} | {state} | {top} | {cent} | {txt} | {semantic} | {notes} |".format(
                name=row["name"],
                collapse=("%.1f" % row["collapse_rate_pct"]) if row["collapse_rate_pct"] is not None else "na",
                coupling=("%.4f" % row["avg_coupling_score"]) if row["avg_coupling_score"] is not None else "na",
                audio=("%.4f" % row["avg_audio_effect"]) if row["avg_audio_effect"] is not None else "na",
                state=("%.4f" % row["avg_state_effect"]) if row["avg_state_effect"] is not None else "na",
                top=("%.2f" % row["avg_top_ratio"]) if row["avg_top_ratio"] is not None else "na",
                cent=("%.0f" % row["avg_cent_var"]) if row["avg_cent_var"] is not None else "na",
                txt=int(bool(row["text_prompt_conditioned"])),
                semantic=row.get("semantic_condition_mode", "none"),
                notes=row["notes"].replace("|", "/"),
            )
        )
    out_md = summaries_root / "hypercube_leaderboard.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    out["json_path"] = str(out_json)
    out["md_path"] = str(out_md)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate hypercube run summaries into a leaderboard.")
    ap.add_argument("--prefix", type=str, default="pathb_hypercube")
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out = build_hypercube_report(repo_root=repo_root, prefix=args.prefix)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
