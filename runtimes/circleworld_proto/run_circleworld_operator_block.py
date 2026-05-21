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

from benchmark_audio_continuation import _load_cases  # noqa: E402
from config import load_config  # noqa: E402
from phase_native_audio_operators import operator_bank_manifest  # noqa: E402
from run_audio_delta_mechanism_probe import (  # noqa: E402
    _future_access_flags,
    _load_circle_cfg,
    _run_case,
    _safe_device,
)
from run_phase_native_audio_selected_route import (  # noqa: E402
    _case_group,
    _json_load,
    _route_dict,
    _route_table,
    _score_selected_rows,
    _select_route,
    _selected_rows_from_cases,
)
from score_phase_native_audio_reentry_guard import _future_access_clean  # noqa: E402


OUTPUT_JSON = "circleworld_operator_block_v1.json"
OUTPUT_MD = "CIRCLEWORLD_OPERATOR_BLOCK_V1.md"
BLOCK_VERSION = "circleworld_operator_block_v1"
DEFAULT_ROUTER_HEAD_ID = "objective_knn5_v1"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_block(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_corr_delta_vs_copy_last": _as_float(score.get("mean_corr_delta_vs_copy_last")),
        "mean_mse_delta_vs_copy_last": _as_float(score.get("mean_mse_delta_vs_copy_last")),
        "mean_loop_delta_vs_copy_last": _as_float(score.get("mean_loop_delta_vs_copy_last")),
        "mean_reentry_delta_vs_copy_last": _as_float(score.get("mean_reentry_delta_vs_copy_last")),
        "mean_harmful_replay_excess_delta_vs_copy_last": score.get("mean_harmful_replay_excess_delta_vs_copy_last"),
        "mean_corr_delta_vs_gain0": score.get("mean_corr_delta_vs_gain0"),
        "strict_target_replay_pass": bool(score.get("strict_target_replay_pass", False)),
    }


def _raw_suite_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _json_load(path)
    score = payload.get("scorecard", {})
    return {
        "path": str(path),
        "schema": payload.get("schema"),
        "status": score.get("status", payload.get("status")),
        "promotion_candidate": bool(score.get("promotion_candidate", payload.get("promotion_candidate", False))),
        "mean_corr_delta_vs_copy_last": _as_float(score.get("mean_circleworld_corr_delta_vs_copy_last")),
        "mean_mse_delta_vs_copy_last": _as_float(score.get("mean_circleworld_mse_delta_vs_copy_last")),
        "mean_loop_delta_vs_copy_last": _as_float(score.get("mean_circleworld_loop_peak_delta_vs_copy_last")),
        "mean_reentry_delta_vs_copy_last": _as_float(score.get("mean_circleworld_reentry_delta_vs_copy_last")),
    }


def _selected_route_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _json_load(path)
    agg = payload.get("score", {}).get("aggregate", {})
    return {
        "path": str(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        **_metric_block(agg),
    }


def _compare_metrics(left: dict[str, Any], right: dict[str, Any] | None) -> dict[str, Any] | None:
    if right is None:
        return None
    keys = [
        "mean_corr_delta_vs_copy_last",
        "mean_mse_delta_vs_copy_last",
        "mean_loop_delta_vs_copy_last",
        "mean_reentry_delta_vs_copy_last",
    ]
    if right.get("mean_harmful_replay_excess_delta_vs_copy_last") is not None:
        keys.append("mean_harmful_replay_excess_delta_vs_copy_last")
    out: dict[str, Any] = {}
    for key in keys:
        if left.get(key) is None or right.get(key) is None:
            continue
        out[f"{key}_minus_reference"] = _as_float(left.get(key)) - _as_float(right.get(key))
    return out


def _beats_raw(block: dict[str, Any], raw: dict[str, Any] | None) -> bool | None:
    if raw is None:
        return None
    return bool(
        _as_float(block.get("mean_corr_delta_vs_copy_last")) > _as_float(raw.get("mean_corr_delta_vs_copy_last"))
        and _as_float(block.get("mean_mse_delta_vs_copy_last")) <= _as_float(raw.get("mean_mse_delta_vs_copy_last"))
        and _as_float(block.get("mean_loop_delta_vs_copy_last")) <= _as_float(raw.get("mean_loop_delta_vs_copy_last"))
        and _as_float(block.get("mean_reentry_delta_vs_copy_last")) <= _as_float(raw.get("mean_reentry_delta_vs_copy_last"))
    )


def _close_to_selected(block: dict[str, Any], selected: dict[str, Any] | None, *, tolerance: float = 1.0e-6) -> bool | None:
    if selected is None:
        return None
    keys = [
        "mean_corr_delta_vs_copy_last",
        "mean_mse_delta_vs_copy_last",
        "mean_loop_delta_vs_copy_last",
        "mean_reentry_delta_vs_copy_last",
        "mean_harmful_replay_excess_delta_vs_copy_last",
    ]
    for key in keys:
        if block.get(key) is None or selected.get(key) is None:
            continue
        if abs(_as_float(block.get(key)) - _as_float(selected.get(key))) > float(tolerance):
            return False
    return True


def select_route_from_policy_payload(
    policy_payload: dict[str, Any],
    case_name: str,
    *,
    route_policy: str,
) -> dict[str, Any]:
    family_routes, global_route, case_routes = _route_table(policy_payload, policy=route_policy)
    return _route_dict(
        _select_route(
            case_name,
            family_routes=family_routes,
            global_route=global_route,
            case_routes=case_routes,
        )
    )


def _markdown(summary: dict[str, Any]) -> str:
    score = summary["score"]["aggregate"]
    comparison = summary.get("comparison", {})
    lines = [
        "# Circleworld Operator Block V1",
        "",
        "This run wraps the current phase-native route-selector evidence as a clean Circleworld operator block.",
        "It is an internal Circleworld architecture consolidation, not a Graduation comparison.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Block version: `{summary['operator_block_version']}`",
        f"- Router head: `{summary['router_head_id']}`",
        f"- Route policy: `{summary['route_policy']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Future access clean: `{summary['future_access_clean']}`",
        "",
        "## Score",
        "",
        "| corr-copy | MSE-copy | loop-copy | reentry-copy | harm-delta | corr-gain0 | strict |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        "| {corr:.9f} | {mse:.9f} | {loop:.9f} | {reentry:.9f} | {harm:.9f} | {cgain:.9f} | `{strict}` |".format(
            corr=_as_float(score.get("mean_corr_delta_vs_copy_last")),
            mse=_as_float(score.get("mean_mse_delta_vs_copy_last")),
            loop=_as_float(score.get("mean_loop_delta_vs_copy_last")),
            reentry=_as_float(score.get("mean_reentry_delta_vs_copy_last")),
            harm=_as_float(score.get("mean_harmful_replay_excess_delta_vs_copy_last")),
            cgain=_as_float(score.get("mean_corr_delta_vs_gain0")),
            strict=bool(score.get("strict_target_replay_pass")),
        ),
        "",
        "## Comparison",
        "",
        f"- Beats raw Circleworld: `{comparison.get('beats_raw_circleworld')}`",
        f"- Close to frozen selected-route artifact: `{comparison.get('close_to_selected_route_reference')}`",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
    ]
    return "\n".join(lines) + "\n"


def run_operator_block(
    *,
    config_path: Path,
    source_route_json: Path,
    suite_json: Path,
    out_dir: Path,
    device_name: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    cases_json: Path | None,
    route_policy: str,
    margin: float,
    router_head_id: str,
    raw_suite_json: Path | None,
    selected_route_json: Path | None,
) -> dict[str, Any]:
    cfg = load_config(str(ROOT / "config.yaml")) if (ROOT / "config.yaml").exists() else load_config()
    sr = int(cfg["data"]["sample_rate"])
    stft_cfg = dict(cfg["data"]["stft"])
    circle_cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    source_policy = _json_load(source_route_json)
    family_routes, global_route, case_routes = _route_table(source_policy, policy=route_policy)

    cases = list(_load_cases(cases_json).items())
    if num_cases > 0:
        cases = cases[:num_cases]
    if not cases:
        raise RuntimeError("No cases selected for Circleworld operator block")

    rendered_cases: list[dict[str, Any]] = []
    route_decisions: list[dict[str, Any]] = []
    for case_name, wav_path in cases:
        route_key = _select_route(
            case_name,
            family_routes=family_routes,
            global_route=global_route,
            case_routes=case_routes,
        )
        mode, mask, mechanism, gain = route_key
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
        route = _route_dict(route_key)
        rendered["selected_route"] = route
        rendered["route_group"] = _case_group(case_name)
        rendered["operator_block_version"] = BLOCK_VERSION
        rendered["router_head_id"] = router_head_id
        rendered_cases.append(rendered)
        route_decisions.append(
            {
                "case_name": case_name,
                "group": _case_group(case_name),
                "source_wav": str(wav_path),
                "route": route,
            }
        )

    rows = [row for case in rendered_cases for row in case.get("rows", [])]
    selected_rows = _selected_rows_from_cases(rendered_cases)
    flags = _future_access_flags(rows)
    block_summary: dict[str, Any] = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_operator_block_v1",
        "operator_block_version": BLOCK_VERSION,
        "router_head_id": router_head_id,
        "router_role": "first_routing_head_inside_circleworld_operator_block_v1",
        "operator_bank": operator_bank_manifest(),
        "config_path": str(config_path),
        "source_route_json": str(source_route_json),
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
        "route_decisions": route_decisions,
        "cases": rendered_cases,
    }
    score = _score_selected_rows(suite_json=suite_json, selected_summary=block_summary, margin=margin)
    block_summary["score"] = score
    block_summary["future_access_clean"] = bool(_future_access_clean([block_summary]))
    block_summary["strict"] = bool(score.get("aggregate", {}).get("strict_target_replay_pass"))

    block_metrics = _metric_block(score.get("aggregate", {}))
    raw_metrics = _raw_suite_metrics(raw_suite_json)
    selected_metrics = _selected_route_metrics(selected_route_json)
    block_summary["comparison"] = {
        "block_metrics": block_metrics,
        "raw_circleworld_reference": raw_metrics,
        "selected_route_reference": selected_metrics,
        "block_minus_raw": _compare_metrics(block_metrics, raw_metrics),
        "block_minus_selected_route": _compare_metrics(block_metrics, selected_metrics),
        "beats_raw_circleworld": _beats_raw(block_metrics, raw_metrics),
        "close_to_selected_route_reference": _close_to_selected(block_metrics, selected_metrics),
    }

    strict = bool(block_summary["strict"])
    if strict and block_summary["future_access_clean"]:
        block_summary["status"] = "operator_block_target_replay_pass"
        block_summary["interpretation"] = (
            "Circleworld operator_block_v1 reproduces the legal selected-route path as an explicit phase-native "
            "operator block. This consolidates the route-selector component into a reusable architecture layer; "
            "it does not promote raw Circleworld and does not compare against protected Graduation RAFA."
        )
    elif not block_summary["future_access_clean"]:
        block_summary["status"] = "operator_block_invalid_future_leakage"
        block_summary["interpretation"] = "Future-access flags failed; this block artifact is invalid."
    else:
        block_summary["status"] = "operator_block_not_yet"
        block_summary["interpretation"] = (
            "Circleworld operator_block_v1 ran, but did not pass the target-normalized replay guard. "
            "Keep it as architecture plumbing only until the route/operator objective improves."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(block_summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(block_summary), encoding="utf-8")
    return block_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Circleworld operator_block_v1 over a selected phase-native route policy.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-route-json", required=True, type=Path)
    parser.add_argument("--suite-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cases-json", default=None, type=Path)
    parser.add_argument("--route-policy", choices=("global", "family_table", "case_table"), default="case_table")
    parser.add_argument("--router-head-id", default=DEFAULT_ROUTER_HEAD_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--prefix-seconds", type=float, default=1.0)
    parser.add_argument("--future-seconds", type=float, default=1.0)
    parser.add_argument("--target-margin", type=float, default=0.01)
    parser.add_argument("--raw-suite-json", default=None, type=Path)
    parser.add_argument("--selected-route-json", default=None, type=Path)
    args = parser.parse_args()
    summary = run_operator_block(
        config_path=args.config,
        source_route_json=args.source_route_json,
        suite_json=args.suite_json,
        out_dir=args.out_dir,
        device_name=args.device,
        num_cases=int(args.num_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        cases_json=args.cases_json,
        route_policy=str(args.route_policy),
        margin=float(args.target_margin),
        router_head_id=str(args.router_head_id),
        raw_suite_json=args.raw_suite_json or args.suite_json,
        selected_route_json=args.selected_route_json,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "strict": bool(summary["score"]["aggregate"].get("strict_target_replay_pass")),
                "future_access_clean": bool(summary.get("future_access_clean")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
