from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_audio_continuation import _load_cases
from run_audio_delta_mechanism_probe import _future_access_flags, _run_case
from score_phase_native_audio_reentry_guard import (
    _collect_delta_rows,
    _copy_last_baseline,
    _future_access_clean,
    _load_required_artifacts,
)
from score_phase_native_audio_target_replay_oracle import (
    _aggregate_selected,
    _augment_rows,
    _copy_last_harm_by_case,
    _target_reentry_by_case,
)
from run_audio_delta_mechanism_probe import _load_circle_cfg, _safe_device
from config import load_config
from profile_registry import case_group as _case_group
from profile_registry import route_dict as _route_dict
from profile_registry import route_key_from_dict as _route_key_from_dict


OUTPUT_JSON = "phase_native_audio_selected_route.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_SELECTED_ROUTE.md"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _route_table(
    source_oracle: dict[str, Any],
    *,
    policy: str,
) -> tuple[
    dict[str, tuple[str, str, str, float]],
    tuple[str, str, str, float] | None,
    dict[str, tuple[str, str, str, float]],
]:
    if policy == "global":
        key = source_oracle.get("global", {}).get("key")
        if not key:
            raise ValueError("Source oracle does not contain a global key")
        return {}, _route_key_from_dict(key), {}
    if policy == "family_table":
        table: dict[str, tuple[str, str, str, float]] = {}
        for group, choice in source_oracle.get("family_choices", {}).items():
            key = choice.get("key", {})
            if key:
                table[str(group)] = _route_key_from_dict(key)
        if not table:
            raise ValueError("Source oracle does not contain family choices")
        return table, None, {}
    if policy == "case_table":
        table = {}
        for case_name, key in source_oracle.get("case_routes", {}).items():
            table[str(case_name)] = _route_key_from_dict(key)
        if not table:
            raise ValueError("Source route artifact does not contain case_routes")
        fallback = source_oracle.get("fallback_route")
        return {}, _route_key_from_dict(fallback) if fallback else None, table
    raise ValueError(f"Unknown route policy: {policy}")


def _select_route(
    case_name: str,
    *,
    family_routes: dict[str, tuple[str, str, str, float]],
    global_route: tuple[str, str, str, float] | None,
    case_routes: dict[str, tuple[str, str, str, float]],
) -> tuple[str, str, str, float]:
    if case_name in case_routes:
        return case_routes[case_name]
    if global_route is not None:
        return global_route
    group = _case_group(case_name)
    if group not in family_routes:
        raise KeyError(f"No family route for group {group!r}")
    return family_routes[group]


def _selected_rows_from_cases(case_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in case_rows:
        route = case.get("selected_route", {})
        key = _route_key_from_dict(route)
        for row in case.get("rows", []):
            if (
                str(row.get("magnitude_mode")) == key[0]
                and str(row.get("mask_mode")) == key[1]
                and str(row.get("mechanism")) == key[2]
                and abs(float(row.get("gain", 0.0)) - key[3]) <= 1.0e-12
            ):
                rows.append(row)
                break
    return rows


def _score_selected_rows(
    *,
    suite_json: Path,
    selected_summary: dict[str, Any],
    margin: float,
) -> dict[str, Any]:
    prefix, _, _ = _load_required_artifacts(suite_json)
    copy_last = _copy_last_baseline(prefix)
    target_reentry = _target_reentry_by_case(prefix)
    copy_last_harm = _copy_last_harm_by_case(prefix, target_reentry, margin=margin)
    flat_rows = _collect_delta_rows(selected_summary, copy_last)
    nonzero_selected = [row for row in flat_rows if abs(float(row.get("gain", 0.0))) > 1.0e-12]
    augmented = _augment_rows(nonzero_selected, target_reentry, copy_last_harm, margin=margin)
    return {
        "copy_last_case_count": len(copy_last),
        "target_reentry_case_count": len(target_reentry),
        "selected_nonzero_row_count": len(augmented),
        "aggregate": _aggregate_selected(augmented, "selected_route"),
    }


def _markdown(summary: dict[str, Any]) -> str:
    score = summary["score"]["aggregate"]
    lines = [
        "# Phase-Native Audio Selected Route",
        "",
        "This run chooses a route before rendering and renders only that selected",
        "phase-only continuation route per case.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Route policy: `{summary['route_policy']}`",
        f"- Source oracle: `{summary['source_oracle_json']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Future access clean: `{summary['future_access_clean']}`",
        "",
        "## Score",
        "",
        "| corr-copy | MSE-copy | loop-copy | reentry-copy | harm-delta | corr-gain0 | strict |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        "| {corr:.9f} | {mse:.9f} | {loop:.9f} | {reentry:.9f} | {harm:.9f} | {cgain:.9f} | `{strict}` |".format(
            corr=float(score.get("mean_corr_delta_vs_copy_last", 0.0)),
            mse=float(score.get("mean_mse_delta_vs_copy_last", 0.0)),
            loop=float(score.get("mean_loop_delta_vs_copy_last", 0.0)),
            reentry=float(score.get("mean_reentry_delta_vs_copy_last", 0.0)),
            harm=float(score.get("mean_harmful_replay_excess_delta_vs_copy_last", 0.0)),
            cgain=float(score.get("mean_corr_delta_vs_gain0", 0.0)),
            strict=bool(score.get("strict_target_replay_pass")),
        ),
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
    ]
    return "\n".join(lines) + "\n"


def run_selected_route(
    *,
    config_path: Path,
    source_oracle_json: Path,
    suite_json: Path,
    out_dir: Path,
    device_name: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    cases_json: Path | None,
    route_policy: str,
    margin: float,
) -> dict[str, Any]:
    cfg = load_config(str(ROOT / "config.yaml")) if (ROOT / "config.yaml").exists() else load_config()
    sr = int(cfg["data"]["sample_rate"])
    stft_cfg = dict(cfg["data"]["stft"])
    circle_cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    source_oracle = _json_load(source_oracle_json)
    family_routes, global_route, case_routes = _route_table(source_oracle, policy=route_policy)

    cases = list(_load_cases(cases_json).items())
    if num_cases > 0:
        cases = cases[:num_cases]
    if not cases:
        raise RuntimeError("No cases selected for selected-route renderer")

    rendered_cases: list[dict[str, Any]] = []
    for case_name, wav_path in cases:
        mode, mask, mechanism, gain = _select_route(
            case_name,
            family_routes=family_routes,
            global_route=global_route,
            case_routes=case_routes,
        )
        rendered = _run_case(
            name=case_name,
            wav_path=wav_path,
            circle_cfg=circle_cfg,
            stft_cfg=stft_cfg,
            sr=sr,
            device=device,
            prefix_seconds=prefix_seconds,
            future_seconds=future_seconds,
            magnitude_modes=[mode],
            mask_modes=[mask],
            mechanisms=[mechanism],
            gains=[0.0, float(gain)],
        )
        rendered["selected_route"] = _route_dict((mode, mask, mechanism, float(gain)))
        rendered["route_group"] = _case_group(case_name)
        rendered_cases.append(rendered)

    rows = [row for case in rendered_cases for row in case.get("rows", [])]
    selected_rows = _selected_rows_from_cases(rendered_cases)
    flags = _future_access_flags(rows)
    selected_summary = {
        "runtime": "circleworld_proto",
        "schema": "phase_native_audio_selected_route_render_v1",
        "config_path": str(config_path),
        "source_oracle_json": str(source_oracle_json),
        "suite_json": str(suite_json),
        "out_dir": str(out_dir),
        "device": str(device),
        "requested_device": str(device_name),
        "sample_rate": sr,
        "prefix_seconds": float(prefix_seconds),
        "future_seconds": float(future_seconds),
        "route_policy": route_policy,
        "case_count": len(rendered_cases),
        "selected_row_count": len(selected_rows),
        "future_target_audio_used_for_metrics_only": True,
        **flags,
        "global_route": _route_dict(global_route) if global_route is not None else None,
        "family_routes": {group: _route_dict(route) for group, route in sorted(family_routes.items())},
        "case_routes": {case: _route_dict(route) for case, route in sorted(case_routes.items())},
        "cases": rendered_cases,
    }
    score = _score_selected_rows(suite_json=suite_json, selected_summary=selected_summary, margin=margin)
    strict = bool(score["aggregate"].get("strict_target_replay_pass"))
    selected_summary["score"] = score
    selected_summary["future_access_clean"] = bool(_future_access_clean([selected_summary]))
    if strict:
        selected_summary["status"] = "selected_route_target_replay_pass"
        selected_summary["interpretation"] = (
            "The route-selected live renderer passes the target-normalized replay guard. "
            "This is stronger than row-oracle evidence because route choice happened before rendering."
        )
    else:
        selected_summary["status"] = "selected_route_not_yet"
        selected_summary["interpretation"] = (
            "The route-selected live renderer does not pass the target-normalized guard. "
            "The route evidence remains an offline scorer result until improved."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(selected_summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(selected_summary), encoding="utf-8")
    return selected_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a preselected phase-native audio route per case.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-oracle-json", required=True, type=Path)
    parser.add_argument("--suite-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cases-json", default=None, type=Path)
    parser.add_argument("--route-policy", choices=("global", "family_table", "case_table"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--prefix-seconds", type=float, default=1.0)
    parser.add_argument("--future-seconds", type=float, default=1.0)
    parser.add_argument("--target-margin", type=float, default=0.01)
    args = parser.parse_args()
    summary = run_selected_route(
        config_path=args.config,
        source_oracle_json=args.source_oracle_json,
        suite_json=args.suite_json,
        out_dir=args.out_dir,
        device_name=args.device,
        num_cases=int(args.num_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        cases_json=args.cases_json,
        route_policy=args.route_policy,
        margin=float(args.target_margin),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "strict": bool(summary["score"]["aggregate"].get("strict_target_replay_pass")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
