from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from score_phase_native_audio_prefix_router_scout import (
    _best_case_rows,
    _case_group,
    _case_rows_by_case,
    _dict_to_key,
    _evaluate_predictions,
)
from score_phase_native_audio_target_replay_oracle import _collect_rows
from score_phase_native_audio_reentry_guard import _key


OUTPUT_JSON = "phase_native_audio_route_transfer.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_ROUTE_TRANSFER.md"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _family_key_from_oracle(oracle: dict[str, Any]) -> dict[str, tuple[str, str, str, float]]:
    out: dict[str, tuple[str, str, str, float]] = {}
    for group, choice in oracle.get("family_choices", {}).items():
        key = choice.get("key", {})
        if key:
            out[str(group)] = _dict_to_key(key)
    return out


def _global_key_from_oracle(oracle: dict[str, Any]) -> tuple[str, str, str, float] | None:
    key = oracle.get("global", {}).get("key")
    return _dict_to_key(key) if key else None


def _key_to_dict(key: tuple[str, str, str, float]) -> dict[str, Any]:
    return {
        "magnitude_mode": key[0],
        "mask_mode": key[1],
        "mechanism": key[2],
        "gain": key[3],
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Route Transfer",
        "",
        "This scorer applies source-lockbox route choices to a target lockbox without",
        "using target future metrics to choose the route.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Future access clean: `{summary['future_access_clean']}`",
        f"- Target case count: `{summary['case_count']}`",
        "",
        "| policy | corr-copy | MSE-copy | loop-copy | harm-delta | corr-gain0 | strict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, row in summary["policies"].items():
        lines.append(
            "| `{label}` | {corr:.9f} | {mse:.9f} | {loop:.9f} | {harm:.9f} | {cgain:.9f} | `{strict}` |".format(
                label=label,
                corr=float(row.get("mean_corr_delta_vs_copy_last", 0.0)),
                mse=float(row.get("mean_mse_delta_vs_copy_last", 0.0)),
                loop=float(row.get("mean_loop_delta_vs_copy_last", 0.0)),
                harm=float(row.get("mean_harmful_replay_excess_delta_vs_copy_last", 0.0)),
                cgain=float(row.get("mean_corr_delta_vs_gain0", 0.0)),
                strict=bool(row.get("strict_target_replay_pass")),
            )
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def score_route_transfer(
    source_oracle_json: Path,
    target_suite_json: Path,
    target_delta_jsons: Sequence[Path],
    out_dir: Path,
    *,
    margin: float,
) -> dict[str, Any]:
    source = _json_load(source_oracle_json)
    meta, rows = _collect_rows(target_suite_json, target_delta_jsons, margin)
    rows_by_case = _case_rows_by_case(rows)
    cases = sorted(rows_by_case)
    target_best_case = _best_case_rows(rows)
    target_case_key = {case: _key(row) for case, row in target_best_case.items()}
    source_family_key = _family_key_from_oracle(source)
    policies: dict[str, dict[str, Any]] = {}

    global_key = _global_key_from_oracle(source)
    if global_key is not None:
        policies["source_global_route"] = _evaluate_predictions(
            label="source_global_route",
            cases=cases,
            predictions={case: global_key for case in cases},
            rows_by_case=rows_by_case,
            family_key=source_family_key,
            case_best_key=target_case_key,
        )

    family_predictions = {
        case: source_family_key[_case_group(case)] for case in cases if _case_group(case) in source_family_key
    }
    policies["source_family_table"] = _evaluate_predictions(
        label="source_family_table",
        cases=cases,
        predictions=family_predictions,
        rows_by_case=rows_by_case,
        family_key=source_family_key,
        case_best_key=target_case_key,
    )

    strict = [label for label, row in policies.items() if bool(row.get("strict_target_replay_pass"))]
    if strict:
        status = "route_transfer_pass"
        interpretation = (
            "At least one source-lockbox route transfers to the target lockbox under the target-normalized guard. "
            "This supports a stable route policy, though live rendering validation is still required."
        )
    else:
        status = "route_transfer_not_yet"
        interpretation = (
            "Source-lockbox routes do not transfer under the strict target-normalized guard. "
            "The route policy is still lockbox-specific or needs richer prefix training."
        )

    summary = {
        "schema": "phase_native_audio_route_transfer_v1",
        "status": status,
        "source_oracle_json": str(source_oracle_json),
        "target_suite_json": str(target_suite_json),
        "target_delta_jsons": [str(path) for path in target_delta_jsons],
        "future_access_clean": bool(meta["future_access_clean"]),
        "target_margin": float(margin),
        "case_count": len(cases),
        "source_global_key": _key_to_dict(global_key) if global_key is not None else None,
        "source_family_keys": {group: _key_to_dict(key) for group, key in sorted(source_family_key.items())},
        "policies": policies,
        "interpretation": interpretation,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer source-lockbox route policies to a target lockbox.")
    parser.add_argument("--source-oracle-json", required=True, type=Path)
    parser.add_argument("--target-suite-json", required=True, type=Path)
    parser.add_argument("--target-delta-json", required=True, action="append", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-margin", type=float, default=0.01)
    args = parser.parse_args()
    summary = score_route_transfer(
        args.source_oracle_json,
        args.target_suite_json,
        args.target_delta_json,
        args.out_dir,
        margin=args.target_margin,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "strict_policies": [
                    label for label, row in summary["policies"].items() if row.get("strict_target_replay_pass")
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
