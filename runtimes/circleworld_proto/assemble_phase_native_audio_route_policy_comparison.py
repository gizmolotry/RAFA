from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OUTPUT_JSON = "phase_native_audio_route_policy_comparison.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_ROUTE_POLICY_COMPARISON.md"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--run must be formatted as label=path")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


def _row(label: str, path: Path) -> dict[str, Any]:
    payload = _json_load(path)
    score = payload.get("score", {}).get("aggregate", {})
    return {
        "label": label,
        "path": str(path),
        "status": payload.get("status"),
        "route_policy": payload.get("route_policy"),
        "source_oracle_json": payload.get("source_oracle_json"),
        "case_count": payload.get("case_count"),
        "mean_corr_delta_vs_copy_last": score.get("mean_corr_delta_vs_copy_last"),
        "median_corr_delta_vs_copy_last": score.get("median_corr_delta_vs_copy_last"),
        "mean_mse_delta_vs_copy_last": score.get("mean_mse_delta_vs_copy_last"),
        "mean_loop_delta_vs_copy_last": score.get("mean_loop_delta_vs_copy_last"),
        "mean_reentry_delta_vs_copy_last": score.get("mean_reentry_delta_vs_copy_last"),
        "mean_harmful_replay_excess_delta_vs_copy_last": score.get(
            "mean_harmful_replay_excess_delta_vs_copy_last"
        ),
        "harmful_replay_win_fraction_vs_copy_last": score.get("harmful_replay_win_fraction_vs_copy_last"),
        "mean_corr_delta_vs_gain0": score.get("mean_corr_delta_vs_gain0"),
        "strict_target_replay_pass": bool(score.get("strict_target_replay_pass")),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):+.9f}"
    if value is None:
        return "NA"
    return str(value)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Route Policy Comparison",
        "",
        f"- Status: `{summary['status']}`",
        f"- Runs: `{len(summary['runs'])}`",
        "",
        "| label | status | corr-copy | MSE-copy | loop-copy | harm-delta | corr-gain0 | strict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["runs"]:
        lines.append(
            "| `{label}` | `{status}` | {corr} | {mse} | {loop} | {harm} | {cgain} | `{strict}` |".format(
                label=row["label"],
                status=row["status"],
                corr=_fmt(row["mean_corr_delta_vs_copy_last"]),
                mse=_fmt(row["mean_mse_delta_vs_copy_last"]),
                loop=_fmt(row["mean_loop_delta_vs_copy_last"]),
                harm=_fmt(row["mean_harmful_replay_excess_delta_vs_copy_last"]),
                cgain=_fmt(row["mean_corr_delta_vs_gain0"]),
                strict=bool(row["strict_target_replay_pass"]),
            )
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def assemble(runs: list[tuple[str, Path]], out_dir: Path) -> dict[str, Any]:
    rows = [_row(label, path) for label, path in runs]
    strict_count = sum(1 for row in rows if row["strict_target_replay_pass"])
    third_rows = [row for row in rows if "third" in row["label"]]
    third_failures = [
        row["label"]
        for row in third_rows
        if not bool(row["strict_target_replay_pass"])
    ]
    third_passes = [
        row["label"]
        for row in third_rows
        if bool(row["strict_target_replay_pass"])
    ]
    if third_failures and third_passes:
        status = "mixed_third_lockbox_route_policy_results"
        interpretation = (
            "Some learned/frozen route policies pass the third holdout and others fail. "
            "This isolates route-family choice as a real control surface rather than a solved classifier."
        )
    elif third_failures:
        status = "learned_router_not_third_lockbox_robust"
        interpretation = (
            "Low-capacity learned routers transfer on original/fresh but do not beat the frozen family table "
            "on the third holdout. The next route learner needs explicit family identification or more training data."
        )
    elif strict_count == len(rows):
        status = "all_route_policies_pass"
        interpretation = "All compared route policies pass the strict target-normalized replay guard."
    else:
        status = "mixed_route_policy_results"
        interpretation = "Some route policies pass and others fail; inspect per-run metrics before promotion."
    summary = {
        "schema": "phase_native_audio_route_policy_comparison_v1",
        "status": status,
        "strict_pass_count": strict_count,
        "run_count": len(rows),
        "runs": rows,
        "interpretation": interpretation,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble selected-route policy comparison.")
    parser.add_argument("--run", action="append", required=True, help="label=path to phase_native_audio_selected_route.json")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = assemble([_parse_run(raw) for raw in args.run], args.out_dir)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "strict_pass_count": summary["strict_pass_count"],
                "run_count": summary["run_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
