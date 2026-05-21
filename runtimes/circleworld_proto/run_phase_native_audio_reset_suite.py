from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"

SUITE_ID = "phase_native_audio_reset_v1"
SCHEMA = "phase_native_audio_reset_suite_v1"
DEFAULT_CONFIG = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "training_run_2026-04-24_agreement_scout_v1"
    / "circleworld_real_anchor_config_cem_v1.json"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "circleworld_proto" / "phase_native_audio_reset_2026-05-17"
OUTPUT_JSON = "phase_native_audio_reset_suite.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_RESET_SUITE.md"


@dataclass(frozen=True)
class SuiteCommand:
    name: str
    description: str
    argv: tuple[str, ...]
    out_dir: Path
    expected_json: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "command": _format_command(self.argv),
            "argv": list(self.argv),
            "out_dir": str(self.out_dir),
            "expected_json": str(self.expected_json),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_command(argv: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in argv])


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values]
    if not clean:
        return 0.0
    return float(sum(clean) / len(clean))


def _default_cases_json() -> Path | None:
    candidates = [
        Path.home()
        / "AppData"
        / "Local"
        / "Temp"
        / "rafa_tokenburst"
        / "audio_predeclared_lockbox_v1_2026_05_06"
        / "audio_predeclared_lockbox_cases.json",
        ROOT
        / "outputs"
        / "circleworld_proto"
        / "audio_predeclared_lockbox_v1_2026-05-06"
        / "audio_predeclared_lockbox_cases.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _optional_path(raw: str | None) -> Path | None:
    if raw is None or not str(raw).strip():
        return None
    return Path(raw).expanduser().absolute()


def _with_common_benchmark_args(
    argv: list[str],
    *,
    config_path: Path,
    out_dir: Path,
    device: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    cases_json: Path | None,
) -> list[str]:
    argv.extend(
        [
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
            "--device",
            device,
            "--num-cases",
            str(num_cases),
            "--prefix-seconds",
            str(prefix_seconds),
            "--future-seconds",
            str(future_seconds),
        ]
    )
    if cases_json is not None:
        argv.extend(["--cases-json", str(cases_json)])
    return argv


def _build_commands(
    *,
    config_path: Path,
    out_dir: Path,
    device: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    cases_json: Path | None,
) -> list[SuiteCommand]:
    commands: list[SuiteCommand] = []
    for name, magnitude_mode in (
        ("continuation_prefix_hold_copyphase", "prefix_hold"),
        ("continuation_flat_copyphase", "flat"),
    ):
        task_dir = out_dir / name
        argv = _with_common_benchmark_args(
            [
                sys.executable,
                str(RUNTIME / "benchmark_audio_continuation.py"),
            ],
            config_path=config_path,
            out_dir=task_dir,
            device=device,
            num_cases=num_cases,
            prefix_seconds=prefix_seconds,
            future_seconds=future_seconds,
            cases_json=cases_json,
        )
        argv.extend(["--magnitude-mode", magnitude_mode, "--phase-seed-policy", "copy_last_waveform_phase"])
        commands.append(
            SuiteCommand(
                name=name,
                description=(
                    "No-future continuation using copy-last waveform phase seed and "
                    f"{magnitude_mode} prefix-derived magnitude carrier."
                ),
                argv=tuple(argv),
                out_dir=task_dir,
                expected_json=task_dir / "audio_continuation_summary.json",
            )
        )

    delta_dir = out_dir / "delta_mechanism_probe_core"
    delta_argv = _with_common_benchmark_args(
        [
            sys.executable,
            str(RUNTIME / "run_audio_delta_mechanism_probe.py"),
        ],
        config_path=config_path,
        out_dir=delta_dir,
        device=device,
        num_cases=num_cases,
        prefix_seconds=prefix_seconds,
        future_seconds=future_seconds,
        cases_json=cases_json,
    )
    delta_argv.extend(
        [
            "--magnitude-modes",
            "prefix_hold,flat",
            "--masks",
            "all_bins,phase_stable_bins,phase_router_bins",
            "--mechanisms",
            "raw,time_smooth_3,phase_velocity_coherent,energy_velocity_time_smooth_3,stable_velocity_time_smooth_3",
            "--gains",
            "0,0.5,1,2",
        ]
    )
    commands.append(
        SuiteCommand(
            name="delta_mechanism_probe_core",
            description=(
                "Fixed prefix-only phase-delta mechanisms over all-bins, stable-phase, "
                "and phase-router masks; scored against each local gain-0 carrier."
            ),
            argv=tuple(delta_argv),
            out_dir=delta_dir,
            expected_json=delta_dir / "audio_delta_mechanism_probe.json",
        )
    )
    return commands


def _run_command(command: SuiteCommand) -> dict[str, Any]:
    command.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = command.out_dir / f"{command.name}.log"
    proc = subprocess.run(
        list(command.argv),
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        "\n".join(
            [
                f"command: {_format_command(command.argv)}",
                f"returncode: {proc.returncode}",
                "",
                "STDOUT",
                proc.stdout,
                "",
                "STDERR",
                proc.stderr,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "name": command.name,
        "status": "ok" if proc.returncode == 0 and command.expected_json.exists() else "error",
        "returncode": int(proc.returncode),
        "log_path": str(log_path),
        "expected_json": str(command.expected_json),
        "expected_json_exists": bool(command.expected_json.exists()),
    }


def _future_access_clean(summary: dict[str, Any]) -> bool:
    return not any(
        bool(summary.get(key))
        for key in (
            "future_target_magnitude_reused",
            "target_future_stft_magnitude_accessed",
            "future_target_phase_reused",
            "target_future_stft_phase_accessed",
        )
    )


def _score_continuation(summary: dict[str, Any]) -> dict[str, Any]:
    methods = summary.get("method_summary", {})
    circle = methods.get("circleworld", {})
    baselines = [
        "baseline_copy_last",
        "baseline_prefix_magnitude_hold",
        "baseline_flat_magnitude",
    ]
    comparisons: dict[str, Any] = {}
    for baseline in baselines:
        item = methods.get(baseline)
        if not item:
            continue
        comparisons[baseline] = {
            "corr_delta": _as_float(circle.get("mean_corr")) - _as_float(item.get("mean_corr")),
            "mae_delta": _as_float(circle.get("mean_mae")) - _as_float(item.get("mean_mae")),
            "mse_delta": _as_float(circle.get("mean_mse")) - _as_float(item.get("mean_mse")),
            "loop_autocorr_peak_delta": _as_float(circle.get("mean_loop_autocorr_peak"))
            - _as_float(item.get("mean_loop_autocorr_peak")),
            "first_chunk_reentry_delta": _as_float(circle.get("mean_first_chunk_reentry"))
            - _as_float(item.get("mean_first_chunk_reentry")),
        }
    return {
        "kind": "audio_continuation",
        "magnitude_mode": summary.get("magnitude_mode"),
        "phase_seed_policy": summary.get("phase_seed_policy"),
        "case_count": int(summary.get("case_count", 0)),
        "future_access_clean": _future_access_clean(summary),
        "circleworld": {
            "mean_corr": _as_float(circle.get("mean_corr")),
            "mean_mae": _as_float(circle.get("mean_mae")),
            "mean_mse": _as_float(circle.get("mean_mse")),
            "mean_loop_autocorr_peak": _as_float(circle.get("mean_loop_autocorr_peak")),
            "mean_first_chunk_reentry": _as_float(circle.get("mean_first_chunk_reentry")),
        },
        "comparisons": comparisons,
    }


def _score_delta_probe(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary.get("aggregate", [])
    nonzero = [row for row in aggregate if abs(_as_float(row.get("gain"))) > 1.0e-12]
    best = None
    if nonzero:
        best = max(nonzero, key=lambda row: _as_float(row.get("mean_target_corr_delta_vs_gain0"), -1.0e9))
    return {
        "kind": "audio_delta_mechanism_probe",
        "status": summary.get("status"),
        "case_count": int(summary.get("case_count", 0)),
        "future_access_clean": _future_access_clean(summary),
        "aggregate_rows": len(aggregate),
        "best_nonzero_corr_delta_row": best,
    }


def _score_existing_outputs(commands: Sequence[SuiteCommand]) -> dict[str, Any]:
    task_scores: list[dict[str, Any]] = []
    missing: list[str] = []
    for command in commands:
        if not command.expected_json.exists():
            missing.append(str(command.expected_json))
            continue
        summary = _json_load(command.expected_json)
        if command.name.startswith("continuation_"):
            task_scores.append({"name": command.name, "artifact": str(command.expected_json), **_score_continuation(summary)})
        elif command.name == "delta_mechanism_probe_core":
            task_scores.append({"name": command.name, "artifact": str(command.expected_json), **_score_delta_probe(summary)})

    continuation = [row for row in task_scores if row.get("kind") == "audio_continuation"]
    copy_deltas = [
        _as_float(row.get("comparisons", {}).get("baseline_copy_last", {}).get("corr_delta"))
        for row in continuation
        if "baseline_copy_last" in row.get("comparisons", {})
    ]
    prefix_deltas = [
        _as_float(row.get("comparisons", {}).get("baseline_prefix_magnitude_hold", {}).get("corr_delta"))
        for row in continuation
        if "baseline_prefix_magnitude_hold" in row.get("comparisons", {})
    ]
    copy_mse_deltas = [
        _as_float(row.get("comparisons", {}).get("baseline_copy_last", {}).get("mse_delta"))
        for row in continuation
        if "baseline_copy_last" in row.get("comparisons", {})
    ]
    copy_loop_deltas = [
        _as_float(row.get("comparisons", {}).get("baseline_copy_last", {}).get("loop_autocorr_peak_delta"))
        for row in continuation
        if "baseline_copy_last" in row.get("comparisons", {})
    ]
    copy_reentry_deltas = [
        _as_float(row.get("comparisons", {}).get("baseline_copy_last", {}).get("first_chunk_reentry_delta"))
        for row in continuation
        if "baseline_copy_last" in row.get("comparisons", {})
    ]
    all_future_clean = bool(task_scores) and all(bool(row.get("future_access_clean")) for row in task_scores)
    promotion_candidate = bool(
        continuation
        and not missing
        and all_future_clean
        and _mean(copy_deltas) > 0.0
        and _mean(prefix_deltas) > 0.0
        and _mean(copy_mse_deltas) <= 0.0
        and _mean(copy_loop_deltas) <= 0.0
        and _mean(copy_reentry_deltas) <= 0.0
    )
    if not task_scores:
        status = "planned_no_results"
    elif promotion_candidate:
        status = "phase_native_audio_promotion_candidate"
    else:
        status = "phase_native_audio_not_promotional"
    return {
        "status": status,
        "promotion_candidate": promotion_candidate,
        "future_access_clean": all_future_clean,
        "missing_expected_outputs": missing,
        "mean_circleworld_corr_delta_vs_copy_last": _mean(copy_deltas),
        "mean_circleworld_corr_delta_vs_prefix_phase_seed": _mean(prefix_deltas),
        "mean_circleworld_mse_delta_vs_copy_last": _mean(copy_mse_deltas),
        "mean_circleworld_loop_peak_delta_vs_copy_last": _mean(copy_loop_deltas),
        "mean_circleworld_reentry_delta_vs_copy_last": _mean(copy_reentry_deltas),
        "task_scores": task_scores,
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    scorecard = summary["scorecard"]
    lines = [
        "# Phase-Native Audio Reset Suite",
        "",
        "This suite is the Circleworld audio-facing reset lane. It tests whether phase-native",
        "law state improves future audio continuation without future magnitude or phase leakage.",
        "",
        "## Run",
        "",
        f"- Suite: `{summary['suite_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Output directory: `{summary['out_dir']}`",
        f"- Cases JSON: `{summary.get('cases_json') or ''}`",
        f"- Device: `{summary['device']}`",
        f"- Cases requested: `{summary['num_cases']}`",
        f"- Prefix/future seconds: `{summary['prefix_seconds']}` / `{summary['future_seconds']}`",
        f"- Executed: `{summary['executed']}`",
        "",
        "## Promotion Contract",
        "",
        "- No future target magnitude or phase reuse.",
        "- Circleworld must beat copy-last waveform and copyphase carrier baselines on correlation.",
        "- MSE, loop autocorrelation, and first-chunk reentry must not worsen versus copy-last.",
        "- Any positive result is still a suite candidate, not a checkpoint promotion, until rerun on the full lockbox.",
        "",
        "## Scorecard",
        "",
        f"- Score status: `{scorecard['status']}`",
        f"- Promotion candidate: `{scorecard['promotion_candidate']}`",
        f"- Future access clean: `{scorecard['future_access_clean']}`",
        f"- Mean corr delta vs copy-last: `{scorecard['mean_circleworld_corr_delta_vs_copy_last']:.9f}`",
        f"- Mean corr delta vs prefix/copyphase carrier: `{scorecard['mean_circleworld_corr_delta_vs_prefix_phase_seed']:.9f}`",
        f"- Mean MSE delta vs copy-last: `{scorecard['mean_circleworld_mse_delta_vs_copy_last']:.9f}`",
        f"- Mean loop-peak delta vs copy-last: `{scorecard['mean_circleworld_loop_peak_delta_vs_copy_last']:.9f}`",
        f"- Mean first-reentry delta vs copy-last: `{scorecard['mean_circleworld_reentry_delta_vs_copy_last']:.9f}`",
        "",
        "## Commands",
        "",
        "| Task | Expected JSON | Command |",
        "| --- | --- | --- |",
    ]
    for command in summary["commands"]:
        lines.append(f"| `{command['name']}` | `{command['expected_json']}` | `{command['command']}` |")
    if summary["results"]:
        lines.extend(["", "## Execution Results", "", "| Task | Status | Return code | Log |", "| --- | --- | ---: | --- |"])
        for row in summary["results"]:
            lines.append(
                f"| `{row['name']}` | `{row['status']}` | `{row['returncode']}` | `{row['log_path']}` |"
            )
    lines.extend(["", "## Interpretation", ""])
    if summary["executed"]:
        lines.extend(
            [
                "This executed suite output is evidence for the configured case set only.",
                "Promotion requires satisfying every contract field above; partial correlation,",
                "MSE, or loop improvements do not override a reentry or leakage failure.",
            ]
        )
    else:
        lines.extend(
            [
                "Dry-run output is a registration artifact only. A real run must be interpreted",
                "against the promotion contract above and written back to the experiment ledger.",
            ]
        )
    return "\n".join(lines) + "\n"


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().absolute()
    out_dir = Path(args.out_dir).expanduser().absolute()
    cases_json = _optional_path(args.cases_json) or _default_cases_json()
    commands = _build_commands(
        config_path=config_path,
        out_dir=out_dir,
        device=str(args.device),
        num_cases=int(args.num_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        cases_json=cases_json,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    execute = bool(args.run) and not bool(args.dry_run)
    results = [_run_command(command) for command in commands] if execute else []
    scorecard = _score_existing_outputs(commands) if execute else {
        "status": "planned_no_results",
        "promotion_candidate": False,
        "future_access_clean": False,
        "missing_expected_outputs": [str(command.expected_json) for command in commands],
        "mean_circleworld_corr_delta_vs_copy_last": 0.0,
        "mean_circleworld_corr_delta_vs_prefix_phase_seed": 0.0,
        "mean_circleworld_mse_delta_vs_copy_last": 0.0,
        "mean_circleworld_loop_peak_delta_vs_copy_last": 0.0,
        "mean_circleworld_reentry_delta_vs_copy_last": 0.0,
        "task_scores": [],
    }
    status = "executed" if execute else "dry_run_planned"
    if execute:
        status = "executed_with_errors" if any(row["status"] != "ok" for row in results) else scorecard["status"]

    summary = {
        "suite_id": SUITE_ID,
        "schema": SCHEMA,
        "created_utc": _now_iso(),
        "status": status,
        "executed": execute,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "cases_json": str(cases_json) if cases_json is not None else None,
        "cases_json_exists": bool(cases_json.exists()) if cases_json is not None else False,
        "device": str(args.device),
        "num_cases": int(args.num_cases),
        "prefix_seconds": float(args.prefix_seconds),
        "future_seconds": float(args.future_seconds),
        "commands": [command.to_json() for command in commands],
        "results": results,
        "scorecard": scorecard,
    }
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Register or run the phase-native Circleworld audio reset suite.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Circleworld JSON config path.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cases-json", default=None, help="Optional predeclared cases JSON. Defaults to known lockbox if present.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--prefix-seconds", type=float, default=1.0)
    ap.add_argument("--future-seconds", type=float, default=1.0)
    ap.add_argument("--run", action="store_true", help="Execute the suite commands. Without this, only write a dry-run spec.")
    ap.add_argument("--dry-run", action="store_true", help="Write the suite spec without executing commands.")
    args = ap.parse_args()
    summary = run_suite(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "suite": summary["suite_id"],
                "json": str(Path(summary["out_dir"]) / OUTPUT_JSON),
                "markdown": str(Path(summary["out_dir"]) / OUTPUT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
