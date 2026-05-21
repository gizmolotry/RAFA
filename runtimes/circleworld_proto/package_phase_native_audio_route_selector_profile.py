from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = "phase_native_audio_route_selector_profile.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_ROUTE_SELECTOR_PROFILE.md"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str | None:
    resolved = _resolve(path)
    if not resolved.exists() or not resolved.is_file():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("labeled paths must be formatted label=path")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


def _artifact(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "sha256": _sha256(path),
    }


def _selected_route_summary(label: str, path: Path) -> dict[str, Any]:
    payload = _json_load(_resolve(path))
    score = payload.get("score", {}).get("aggregate", {})
    return {
        "label": label,
        "artifact": _artifact(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "route_policy": payload.get("route_policy"),
        "source_oracle_json": payload.get("source_oracle_json"),
        "suite_json": payload.get("suite_json"),
        "case_count": payload.get("case_count"),
        "selected_row_count": payload.get("selected_row_count"),
        "future_access_clean": bool(payload.get("future_access_clean")),
        "future_target_magnitude_reused": bool(payload.get("future_target_magnitude_reused")),
        "target_future_stft_magnitude_accessed": bool(payload.get("target_future_stft_magnitude_accessed")),
        "future_target_phase_reused": bool(payload.get("future_target_phase_reused")),
        "target_future_stft_phase_accessed": bool(payload.get("target_future_stft_phase_accessed")),
        "strict_target_replay_pass": bool(score.get("strict_target_replay_pass")),
        "metrics": {
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
        },
    }


def _raw_suite_summary(label: str, path: Path) -> dict[str, Any]:
    payload = _json_load(_resolve(path))
    scorecard = payload.get("scorecard", {})
    return {
        "label": label,
        "artifact": _artifact(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "executed": bool(payload.get("executed")),
        "future_access_clean": bool(scorecard.get("future_access_clean")),
        "promotion_candidate": bool(scorecard.get("promotion_candidate")),
        "metrics": {
            "mean_circleworld_corr_delta_vs_copy_last": scorecard.get(
                "mean_circleworld_corr_delta_vs_copy_last"
            ),
            "mean_circleworld_mse_delta_vs_copy_last": scorecard.get(
                "mean_circleworld_mse_delta_vs_copy_last"
            ),
            "mean_circleworld_loop_peak_delta_vs_copy_last": scorecard.get(
                "mean_circleworld_loop_peak_delta_vs_copy_last"
            ),
            "mean_circleworld_reentry_delta_vs_copy_last": scorecard.get(
                "mean_circleworld_reentry_delta_vs_copy_last"
            ),
        },
    }


def _policy_summary(path: Path) -> dict[str, Any]:
    payload = _json_load(_resolve(path))
    return {
        "artifact": _artifact(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "model": payload.get("model"),
        "feature_count": payload.get("feature_count"),
        "feature_keys_sha256": hashlib.sha256(
            json.dumps(payload.get("feature_keys", []), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "train_case_count": payload.get("train_case_count"),
        "target_case_count": payload.get("target_case_count"),
        "train_sets": payload.get("train_sets", []),
        "target_delta_json": payload.get("target_delta_json"),
        "target_margin": payload.get("target_margin"),
        "training_diagnostics": payload.get("training_diagnostics", {}),
        "source_future_metrics_used_for_route_training_labels": bool(
            payload.get("source_future_metrics_used_for_route_training_labels")
        ),
        "target_future_audio_used_for_route_selection": bool(
            payload.get("target_future_audio_used_for_route_selection")
        ),
        "target_future_metrics_used_for_route_selection": bool(
            payload.get("target_future_metrics_used_for_route_selection")
        ),
        "predicted_route_counts": payload.get("predicted_route_counts", {}),
        "fallback_route": payload.get("fallback_route", {}),
    }


def _profile_status(profile: dict[str, Any]) -> str:
    selected = profile["selected_route_validations"]
    raw = profile["raw_circleworld_context"]
    selected_ok = bool(selected) and all(
        row["future_access_clean"]
        and row["strict_target_replay_pass"]
        and not row["future_target_magnitude_reused"]
        and not row["target_future_stft_magnitude_accessed"]
        and not row["future_target_phase_reused"]
        and not row["target_future_stft_phase_accessed"]
        for row in selected
    )
    raw_guard_ok = bool(raw) and all(
        row["future_access_clean"] and not row["promotion_candidate"] for row in raw
    )
    policy = profile["primary_policy"]
    policy_ok = (
        policy["status"] == "objective_route_policy_ready"
        and policy["model"] == profile["component_id"]
        and not policy["target_future_audio_used_for_route_selection"]
        and not policy["target_future_metrics_used_for_route_selection"]
    )
    if selected_ok and raw_guard_ok and policy_ok:
        return "route_selector_component_candidate_frozen"
    return "route_selector_profile_frozen_with_warnings"


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):+.9f}"
    if value is None:
        return "NA"
    return str(value)


def _markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Route Selector Profile",
        "",
        f"- Status: `{profile['status']}`",
        f"- Component: `{profile['component_id']}`",
        f"- Scope: `{profile['promotion_scope']}`",
        f"- Primary policy: `{profile['primary_policy']['artifact']['path']}`",
        f"- Feature count: `{profile['primary_policy']['feature_count']}`",
        f"- Training cases: `{profile['primary_policy']['train_case_count']}`",
        f"- Target cases: `{profile['primary_policy']['target_case_count']}`",
        "",
        "## Selected-Route Evidence",
        "",
        "| label | status | corr-copy | MSE-copy | loop-copy | harm-delta | corr-gain0 | strict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in profile["selected_route_validations"]:
        metrics = row["metrics"]
        lines.append(
            "| `{label}` | `{status}` | {corr} | {mse} | {loop} | {harm} | {cgain} | `{strict}` |".format(
                label=row["label"],
                status=row["status"],
                corr=_fmt(metrics.get("mean_corr_delta_vs_copy_last")),
                mse=_fmt(metrics.get("mean_mse_delta_vs_copy_last")),
                loop=_fmt(metrics.get("mean_loop_delta_vs_copy_last")),
                harm=_fmt(metrics.get("mean_harmful_replay_excess_delta_vs_copy_last")),
                cgain=_fmt(metrics.get("mean_corr_delta_vs_gain0")),
                strict=row["strict_target_replay_pass"],
            )
        )
    lines.extend(
        [
            "",
            "## Raw Circleworld Guard",
            "",
            "| label | status | promotional | corr-copy | MSE-copy | loop-copy | reentry-copy |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in profile["raw_circleworld_context"]:
        metrics = row["metrics"]
        lines.append(
            "| `{label}` | `{status}` | `{promo}` | {corr} | {mse} | {loop} | {reentry} |".format(
                label=row["label"],
                status=row["status"],
                promo=row["promotion_candidate"],
                corr=_fmt(metrics.get("mean_circleworld_corr_delta_vs_copy_last")),
                mse=_fmt(metrics.get("mean_circleworld_mse_delta_vs_copy_last")),
                loop=_fmt(metrics.get("mean_circleworld_loop_peak_delta_vs_copy_last")),
                reentry=_fmt(metrics.get("mean_circleworld_reentry_delta_vs_copy_last")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            profile["interpretation"],
        ]
    )
    return "\n".join(lines) + "\n"


def package_profile(
    *,
    component_id: str,
    primary_policy_json: Path,
    selected_routes: list[tuple[str, Path]],
    raw_suites: list[tuple[str, Path]],
    out_dir: Path,
) -> dict[str, Any]:
    profile = {
        "schema": "phase_native_audio_route_selector_profile_v1",
        "component_id": component_id,
        "claim": (
            "A no-future objective-aware route selector over phase-only Circleworld continuation "
            "operators improves copy-last-relative continuation metrics on held-out lockboxes."
        ),
        "promotion_scope": "route_selector_component_only_not_full_audio_model",
        "primary_policy": _policy_summary(primary_policy_json),
        "selected_route_validations": [
            _selected_route_summary(label, path) for label, path in selected_routes
        ],
        "raw_circleworld_context": [_raw_suite_summary(label, path) for label, path in raw_suites],
        "contract": {
            "target_route_selection_uses_future_audio": False,
            "target_route_selection_uses_future_metrics": False,
            "source_future_metrics_used_for_training_labels": True,
            "raw_circleworld_must_remain_separate_from_component_claim": True,
        },
        "interpretation": (
            "This freezes objective_knn5_v1 as a component candidate. It does not promote raw "
            "Circleworld or declare a full audio model win; it freezes the narrower route-selector "
            "claim so future comparisons can audit source/target separation and no-future selection."
        ),
    }
    profile["status"] = _profile_status(profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(profile, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(profile), encoding="utf-8")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a phase-native audio route-selector profile.")
    parser.add_argument("--component-id", default="objective_knn5_v1")
    parser.add_argument("--primary-policy-json", required=True, type=Path)
    parser.add_argument("--selected-route", action="append", required=True, help="label=path")
    parser.add_argument("--raw-suite", action="append", required=True, help="label=path")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    profile = package_profile(
        component_id=str(args.component_id),
        primary_policy_json=args.primary_policy_json,
        selected_routes=[_parse_labeled_path(raw) for raw in args.selected_route],
        raw_suites=[_parse_labeled_path(raw) for raw in args.raw_suite],
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                "status": profile["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "selected_route_count": len(profile["selected_route_validations"]),
                "raw_suite_count": len(profile["raw_circleworld_context"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
