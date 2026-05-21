from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = "phase_native_audio_route_selector_contract_audit.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_ROUTE_SELECTOR_CONTRACT_AUDIT.md"
_EXTRA_RESOLUTION_ROOTS: list[Path] = []

FORBIDDEN_FEATURE_SUBSTRINGS = (
    "target_corr",
    "target_mae",
    "target_mse",
    "target_rms",
    "output_rms",
    "vs_gain0",
    "loop_autocorr",
    "first_chunk_reentry",
    "harmful",
    "copy_last",
    "future_target",
    "target_future",
    "row_objective",
    "label_metrics",
    "selected_route",
)
ALLOWED_FUTURE_FEATURE_KEYS = {"case.future_samples", "case.future_stft_frames"}


def _add_resolution_root(candidate: Path) -> None:
    resolved = candidate.resolve()
    if resolved != ROOT and resolved not in _EXTRA_RESOLUTION_ROOTS:
        _EXTRA_RESOLUTION_ROOTS.append(resolved)


def _configure_resolution_roots(profile_json: Path) -> None:
    _EXTRA_RESOLUTION_ROOTS.clear()
    if profile_json.is_absolute():
        parts_lower = [part.lower() for part in profile_json.parts]
        for index, part in enumerate(parts_lower):
            if part == "outputs" and index > 0:
                _add_resolution_root(Path(*profile_json.parts[:index]))
                break
    _add_resolution_root(Path.cwd())


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [ROOT, *_EXTRA_RESOLUTION_ROOTS]
    for root in candidates:
        resolved = root / path
        if resolved.exists():
            return resolved
    return ROOT / path


def _json_load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8-sig"))


def _norm(path: str | Path | None) -> str:
    if path is None:
        return ""
    return str(_resolve(path).resolve()).lower()


def _suite_root(path: str | Path | None) -> str:
    if path is None:
        return ""
    resolved = _resolve(path).resolve()
    if resolved.name == "phase_native_audio_reset_suite.json":
        return str(resolved.parent).lower()
    if resolved.name == "audio_delta_mechanism_probe.json" and resolved.parent.name == "delta_mechanism_probe_core":
        return str(resolved.parent.parent).lower()
    return str(resolved.parent).lower()


def _check(checks: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details or {}})


def _policy_feature_check(policy: dict[str, Any]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for key in policy.get("feature_keys", []):
        key_s = str(key)
        if not (key_s.startswith("case.") or key_s.startswith("circleworld_meta.")):
            bad.append(key_s)
            continue
        if "future" in key_s and key_s not in ALLOWED_FUTURE_FEATURE_KEYS:
            bad.append(key_s)
            continue
        if any(token in key_s for token in FORBIDDEN_FEATURE_SUBSTRINGS):
            bad.append(key_s)
    return not bad, bad


def _score_guard_ok(score: dict[str, Any]) -> bool:
    return (
        bool(score.get("strict_target_replay_pass"))
        and float(score.get("mean_corr_delta_vs_copy_last", 0.0)) > 0.0
        and float(score.get("mean_mse_delta_vs_copy_last", 0.0)) < 0.0
        and float(score.get("mean_loop_delta_vs_copy_last", 0.0)) < 0.0
        and float(score.get("mean_harmful_replay_excess_delta_vs_copy_last", 0.0)) < 0.0
        and float(score.get("mean_corr_delta_vs_gain0", 0.0)) > 0.0
    )


def _audit_policy(policy_path: str, selected_suite_json: str | None, checks: list[dict[str, Any]]) -> dict[str, Any]:
    policy = _json_load(policy_path)
    target_root = _suite_root(policy.get("target_delta_json"))
    train_suite_roots = [_suite_root(row.get("suite_json")) for row in policy.get("train_sets", [])]
    train_delta_paths = [
        _norm(delta)
        for row in policy.get("train_sets", [])
        for delta in row.get("delta_jsons", [])
    ]
    feature_ok, bad_features = _policy_feature_check(policy)
    selected_root = _suite_root(selected_suite_json) if selected_suite_json else ""

    _check(
        checks,
        f"policy_schema::{policy_path}",
        policy.get("schema") == "phase_native_audio_objective_route_policy_v1"
        and policy.get("status") == "objective_route_policy_ready",
        {"schema": policy.get("schema"), "status": policy.get("status")},
    )
    _check(
        checks,
        f"policy_target_selection_no_future::{policy_path}",
        not bool(policy.get("target_future_audio_used_for_route_selection"))
        and not bool(policy.get("target_future_metrics_used_for_route_selection")),
        {
            "target_future_audio_used_for_route_selection": policy.get(
                "target_future_audio_used_for_route_selection"
            ),
            "target_future_metrics_used_for_route_selection": policy.get(
                "target_future_metrics_used_for_route_selection"
            ),
        },
    )
    _check(
        checks,
        f"policy_source_labels_explicit::{policy_path}",
        bool(policy.get("source_future_metrics_used_for_route_training_labels")),
        {"source_future_metrics_used_for_route_training_labels": policy.get("source_future_metrics_used_for_route_training_labels")},
    )
    _check(
        checks,
        f"policy_feature_keys_no_future_metrics::{policy_path}",
        feature_ok,
        {"bad_feature_keys": bad_features[:25], "bad_feature_key_count": len(bad_features)},
    )
    _check(
        checks,
        f"policy_feature_count_consistent::{policy_path}",
        int(policy.get("feature_count", -1)) == len(policy.get("feature_keys", [])),
        {"feature_count": policy.get("feature_count"), "actual": len(policy.get("feature_keys", []))},
    )
    _check(
        checks,
        f"policy_target_not_in_training::{policy_path}",
        target_root not in train_suite_roots and _norm(policy.get("target_delta_json")) not in train_delta_paths,
        {
            "target_root": target_root,
            "train_suite_roots": train_suite_roots,
            "target_delta_json": policy.get("target_delta_json"),
        },
    )
    if selected_root:
        _check(
            checks,
            f"policy_target_matches_selected_suite::{policy_path}",
            target_root == selected_root,
            {"policy_target_root": target_root, "selected_suite_root": selected_root},
        )

    return {
        "path": policy_path,
        "model": policy.get("model"),
        "train_case_count": policy.get("train_case_count"),
        "target_case_count": policy.get("target_case_count"),
        "feature_count": policy.get("feature_count"),
        "target_root": target_root,
        "train_suite_roots": train_suite_roots,
    }


def _audit_selected_route(row: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = row.get("artifact", {}).get("path")
    selected = _json_load(path)
    score = selected.get("score", {}).get("aggregate", {})
    policy_path = str(selected.get("source_oracle_json"))
    policy = _json_load(policy_path)
    target_case_count = int(policy.get("target_case_count", -1))

    _check(
        checks,
        f"selected_schema::{path}",
        selected.get("schema") == "phase_native_audio_selected_route_render_v1",
        {"schema": selected.get("schema")},
    )
    _check(
        checks,
        f"selected_future_access_clean::{path}",
        bool(selected.get("future_access_clean"))
        and not bool(selected.get("future_target_magnitude_reused"))
        and not bool(selected.get("target_future_stft_magnitude_accessed"))
        and not bool(selected.get("future_target_phase_reused"))
        and not bool(selected.get("target_future_stft_phase_accessed")),
        {
            "future_access_clean": selected.get("future_access_clean"),
            "future_target_magnitude_reused": selected.get("future_target_magnitude_reused"),
            "target_future_stft_magnitude_accessed": selected.get("target_future_stft_magnitude_accessed"),
            "future_target_phase_reused": selected.get("future_target_phase_reused"),
            "target_future_stft_phase_accessed": selected.get("target_future_stft_phase_accessed"),
        },
    )
    _check(
        checks,
        f"selected_case_count_matches_policy::{path}",
        int(selected.get("case_count", -2)) == target_case_count
        and int(selected.get("selected_row_count", -3)) == target_case_count,
        {
            "selected_case_count": selected.get("case_count"),
            "selected_row_count": selected.get("selected_row_count"),
            "policy_target_case_count": target_case_count,
        },
    )
    _check(
        checks,
        f"selected_metric_guard::{path}",
        _score_guard_ok(score),
        {
            "strict_target_replay_pass": score.get("strict_target_replay_pass"),
            "mean_corr_delta_vs_copy_last": score.get("mean_corr_delta_vs_copy_last"),
            "mean_mse_delta_vs_copy_last": score.get("mean_mse_delta_vs_copy_last"),
            "mean_loop_delta_vs_copy_last": score.get("mean_loop_delta_vs_copy_last"),
            "mean_harmful_replay_excess_delta_vs_copy_last": score.get(
                "mean_harmful_replay_excess_delta_vs_copy_last"
            ),
            "mean_corr_delta_vs_gain0": score.get("mean_corr_delta_vs_gain0"),
        },
    )
    _audit_policy(policy_path, selected.get("suite_json"), checks)

    return {
        "label": row.get("label"),
        "path": path,
        "policy_path": policy_path,
        "status": selected.get("status"),
        "route_policy": selected.get("route_policy"),
        "case_count": selected.get("case_count"),
        "strict_target_replay_pass": score.get("strict_target_replay_pass"),
        "mean_corr_delta_vs_copy_last": score.get("mean_corr_delta_vs_copy_last"),
        "mean_mse_delta_vs_copy_last": score.get("mean_mse_delta_vs_copy_last"),
        "mean_loop_delta_vs_copy_last": score.get("mean_loop_delta_vs_copy_last"),
        "mean_harmful_replay_excess_delta_vs_copy_last": score.get(
            "mean_harmful_replay_excess_delta_vs_copy_last"
        ),
    }


def _audit_raw_suite(row: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = row.get("artifact", {}).get("path")
    raw = _json_load(path)
    scorecard = raw.get("scorecard", {})
    _check(
        checks,
        f"raw_suite_future_access_clean::{path}",
        bool(scorecard.get("future_access_clean")),
        {"future_access_clean": scorecard.get("future_access_clean")},
    )
    _check(
        checks,
        f"raw_suite_not_promotional::{path}",
        raw.get("status") == "phase_native_audio_not_promotional"
        and not bool(scorecard.get("promotion_candidate")),
        {"status": raw.get("status"), "promotion_candidate": scorecard.get("promotion_candidate")},
    )
    return {
        "label": row.get("label"),
        "path": path,
        "status": raw.get("status"),
        "promotion_candidate": scorecard.get("promotion_candidate"),
        "mean_circleworld_corr_delta_vs_copy_last": scorecard.get("mean_circleworld_corr_delta_vs_copy_last"),
        "mean_circleworld_reentry_delta_vs_copy_last": scorecard.get("mean_circleworld_reentry_delta_vs_copy_last"),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Route Selector Contract Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Checks passed: `{summary['passed_count']}` / `{summary['check_count']}`",
        f"- Profile: `{summary['profile_json']}`",
        "",
        "## Selected-Route Audits",
        "",
        "| label | status | corr-copy | MSE-copy | loop-copy | harm-delta | strict |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["selected_route_audits"]:
        lines.append(
            "| `{label}` | `{status}` | {corr:+.9f} | {mse:+.9f} | {loop:+.9f} | {harm:+.9f} | `{strict}` |".format(
                label=row["label"],
                status=row["status"],
                corr=float(row.get("mean_corr_delta_vs_copy_last") or 0.0),
                mse=float(row.get("mean_mse_delta_vs_copy_last") or 0.0),
                loop=float(row.get("mean_loop_delta_vs_copy_last") or 0.0),
                harm=float(row.get("mean_harmful_replay_excess_delta_vs_copy_last") or 0.0),
                strict=bool(row.get("strict_target_replay_pass")),
            )
        )
    lines.extend(["", "## Failed Checks", ""])
    failures = [check for check in summary["checks"] if not check["passed"]]
    if not failures:
        lines.append("None.")
    else:
        for check in failures:
            lines.append(f"- `{check['name']}`: `{json.dumps(check['details'], sort_keys=True)}`")
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def audit_profile(*, profile_json: Path, out_dir: Path) -> dict[str, Any]:
    _configure_resolution_roots(profile_json)
    profile = _json_load(profile_json)
    checks: list[dict[str, Any]] = []
    selected_audits = [
        _audit_selected_route(row, checks) for row in profile.get("selected_route_validations", [])
    ]
    raw_audits = [_audit_raw_suite(row, checks) for row in profile.get("raw_circleworld_context", [])]
    primary_policy = _audit_policy(profile.get("primary_policy", {}).get("artifact", {}).get("path", ""), None, checks)
    _check(
        checks,
        "profile_scope_component_only",
        profile.get("promotion_scope") == "route_selector_component_only_not_full_audio_model",
        {"promotion_scope": profile.get("promotion_scope")},
    )
    passed = sum(1 for check in checks if check["passed"])
    status = "route_selector_contract_pass" if passed == len(checks) else "route_selector_contract_fail"
    summary = {
        "schema": "phase_native_audio_route_selector_contract_audit_v1",
        "status": status,
        "profile_json": str(profile_json),
        "check_count": len(checks),
        "passed_count": passed,
        "failed_count": len(checks) - passed,
        "checks": checks,
        "primary_policy_audit": primary_policy,
        "selected_route_audits": selected_audits,
        "raw_suite_audits": raw_audits,
        "interpretation": (
            "The frozen route selector satisfies the no-future target-selection, source/target "
            "separation, selected-route metric, and raw-Circleworld separation contracts."
            if status == "route_selector_contract_pass"
            else "At least one route-selector contract check failed; inspect failed checks before promotion language."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a frozen phase-native route selector profile.")
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = audit_profile(profile_json=args.profile_json, out_dir=args.out_dir)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "passed_count": summary["passed_count"],
                "check_count": summary["check_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
