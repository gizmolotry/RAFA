from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_audio_continuity import benchmark_folder
from benchmark_circleworld_real_anchor import _load_cases_json, run_benchmark
from build_law_token_library import build_library
from evaluate_circleworld import evaluate_config
from train_circleworld_real_anchor import train_circleworld_real_anchor


DEFAULT_BASE_CONFIG = ROOT / "checkpoints_circleworld_proto" / "training_run_2026-04-17_native_multimode_masked_v1" / "circleworld_real_anchor_config_cem_v1.json"
DEFAULT_EXPANDED_CASES = ROOT / "outputs" / "circleworld_proto" / "expanded_anchor_cases_2026-04-09.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _real_audio_score(block: dict[str, Any]) -> float:
    return float(block["mean_corr"] - 0.5 * block["mean_mae"] - 0.25 * block["mean_mse"])


def _branch_score(heldout: dict[str, Any]) -> float:
    return float(
        2.0 * heldout.get("mean_real_branch_fraction", 0.0)
        + 0.75 * heldout.get("mean_slot2_live_fraction", 0.0)
        + 1.5 * heldout.get("mean_relation_handoff_drive", 0.0)
        + 0.5 * heldout.get("mean_relation_attn_10", 0.0)
        + 0.5 * heldout.get("mean_branch_positive_mask", 0.0)
        - 0.5 * heldout.get("mean_branch_negative_mask", 0.0)
        - 0.5 * heldout.get("mean_silent_singlepath_fraction", 0.0)
    )


def _overall_score(benchmark: dict[str, Any], heldout: dict[str, Any], nested: dict[str, Any]) -> float:
    nested_bonus = 1.0 if nested.get("overall_read") == "nested_commitment_signal_present" else 0.0
    return float(
        0.55 * _real_audio_score(benchmark)
        + 0.35 * _branch_score(heldout)
        + 0.10 * nested_bonus
    )


def _variant_init_payload(base_payload: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(base_payload)
    cfg = payload["config"] if "config" in payload else payload
    cfg.update(overrides)
    return payload


def _default_score_cfg() -> dict[str, float]:
    return {
        "target_corr": 0.77,
        "target_mae": 0.088,
        "target_dom": 0.62,
        "target_entropy": 0.58,
        "min_promotions": 8.0,
        "w_corr": 0.7,
        "w_mae": 2.2,
        "w_dom": 0.3,
        "w_entropy": 0.3,
        "w_promote": 0.05,
        "w_residue": 0.16,
        "w_promotability": 0.08,
        "w_major": 0.08,
        "w_prefix_alignment": 0.35,
        "w_prefix_delta": 0.20,
        "corr_low": 0.55,
        "corr_high": 0.92,
        "mae_low": 0.045,
        "mae_high": 0.125,
        "w_corr_band": 1.5,
        "w_mae_band": 2.4,
        "w_baseline_corr": 3.0,
        "w_baseline_mae": 3.0,
        "baseline_corr_margin": 0.01,
        "baseline_mae_margin": 0.002,
        "w_real_branch": 2.0,
        "w_meso_branch": 45000.0,
        "w_slot2_live": 0.3,
        "w_silent_singlepath": 0.25,
        "w_branch_positive_mask": 2.5,
        "w_branch_negative_mask": 0.75,
        "w_decorative_slot2": 1.25,
        "w_relation_handoff": 4.0,
        "w_relation_attn_10": 0.75,
    }


def _default_case_weights() -> dict[str, float]:
    return {
        "clarinet": 2.5,
        "metal-clang": 2.2,
        "mystic-chanting": 1.7,
        "anvil-impact": 1.5,
    }


def _variant_specs() -> list[dict[str, Any]]:
    relation_seed = {
        "branching_mode": "native_multimode",
        "branch_law_version": "relational_qkv_v2",
        "aux_mask_suppression": 0.22,
        "split_support_gain": 0.68,
        "relation_attention_gain": 0.78,
        "relation_attention_sharpness": 1.35,
        "relation_value_gain": 0.42,
        "relation_support_gain": 0.34,
        "relation_logit_gain": 0.30,
        "relation_qtrace_gain": 0.20,
        "relation_residual_mix": 0.72,
    }
    return [
        {
            "name": "parametric_masked_control",
            "train": False,
            "overrides": {},
        },
        {
            "name": "relational_qkv_v2_ramanujan",
            "train": True,
            "overrides": {**relation_seed, "branch_kernel_version": "ramanujan"},
        },
        {
            "name": "relational_qkv_v2_phase_only",
            "train": True,
            "overrides": {**relation_seed, "branch_kernel_version": "phase_only"},
        },
        {
            "name": "relational_qkv_v2_qtrace_only",
            "train": True,
            "overrides": {**relation_seed, "branch_kernel_version": "qtrace_only"},
        },
        {
            "name": "relational_qkv_v2_uniform",
            "train": True,
            "overrides": {**relation_seed, "branch_kernel_version": "uniform"},
        },
    ]


def _run_nested_commitment(
    config_path: Path,
    out_dir: Path,
    device_name: str,
    case_json: Path,
    mode: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(RUNTIME / "test_nested_commitment.py"),
        "--config",
        str(config_path),
        "--out-dir",
        str(out_dir),
        "--device",
        device_name,
        "--depth",
        "3",
        "--fork-depth",
        "1",
        "--mode",
        mode,
        "--case-json",
        str(case_json),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return _load_json(out_dir / "nested_commitment_report.json")


def run_series(
    out_dir: Path,
    base_config_path: Path,
    expanded_cases_path: Path,
    device_name: str,
    iterations: int,
    population: int,
    elite_count: int,
    clip_seconds: int,
    phase_blend: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_payload = _load_json(base_config_path)
    expanded_cases = _load_cases_json(expanded_cases_path)
    score_cfg = _default_score_cfg()
    case_weights = _default_case_weights()

    rows: list[dict[str, Any]] = []
    for idx, variant in enumerate(_variant_specs()):
        variant_dir = out_dir / variant["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        init_config_path = variant_dir / "init_config.json"
        init_payload = _variant_init_payload(base_payload, variant["overrides"])
        _write_json(init_config_path, init_payload)
        config_path = init_config_path
        train_summary = None

        if variant["train"]:
            train_summary = train_circleworld_real_anchor(
                out_dir=variant_dir / "training",
                checkpoint_dir=variant_dir / "checkpoint",
                iterations=iterations,
                population=population,
                elite_count=elite_count,
                seed=20260417 + idx,
                device_name=device_name,
                clip_seconds=clip_seconds,
                phase_blend=phase_blend,
                init_config_path=init_config_path,
                score_cfg=score_cfg,
                case_weights=case_weights,
            )
            config_path = Path(train_summary["checkpoint"])

        heldout = evaluate_config(
            config_path=config_path,
            out_dir=variant_dir / "heldout_eval",
            time_steps=128,
            device_name=device_name,
        )
        benchmark = run_benchmark(
            config_path=config_path,
            out_dir=variant_dir / "benchmark_expanded",
            device_name=device_name,
            clip_seconds=clip_seconds,
            phase_blend=phase_blend,
            rerender_check=True,
            cases=expanded_cases,
        )
        continuity_circleworld = benchmark_folder(
            folder=variant_dir / "benchmark_expanded",
            pattern="*_circleworld.wav",
            out_path=variant_dir / "benchmark_expanded" / "continuity_circleworld.json",
            min_loop_seconds=0.5,
            max_loop_seconds=4.0,
            chunk_seconds=2.0,
        )
        continuity_reference = benchmark_folder(
            folder=variant_dir / "benchmark_expanded",
            pattern="*_reference.wav",
            out_path=variant_dir / "benchmark_expanded" / "continuity_reference.json",
            min_loop_seconds=0.5,
            max_loop_seconds=4.0,
            chunk_seconds=2.0,
        )
        nested = _run_nested_commitment(
            config_path=config_path,
            out_dir=variant_dir / "nested_commitment",
            device_name=device_name,
            case_json=expanded_cases_path,
            mode="native_multimode",
        )
        law_tokens = build_library(
            config_path=config_path,
            out_dir=variant_dir / "law_token_library",
            cases_path=expanded_cases_path,
            device_name=device_name,
            clip_seconds=clip_seconds,
        )

        row = {
            "variant": variant["name"],
            "trained": bool(variant["train"]),
            "config_path": str(config_path),
            "train_summary": str((variant_dir / "training" / "train_summary.json")) if variant["train"] else None,
            "heldout_summary": str(variant_dir / "heldout_eval" / "heldout_summary.json"),
            "benchmark_summary": str(variant_dir / "benchmark_expanded" / "benchmark_summary.json"),
            "nested_summary": str(variant_dir / "nested_commitment" / "nested_commitment_report.json"),
            "law_token_library": str(variant_dir / "law_token_library" / "law_token_library.json"),
            "continuity_circleworld": str(variant_dir / "benchmark_expanded" / "continuity_circleworld.json"),
            "continuity_reference": str(variant_dir / "benchmark_expanded" / "continuity_reference.json"),
            "benchmark_mean_corr": benchmark["mean_corr"],
            "benchmark_mean_mae": benchmark["mean_mae"],
            "benchmark_mean_mse": benchmark["mean_mse"],
            "benchmark_real_audio_score": _real_audio_score(benchmark),
            "heldout_mean_real_branch_fraction": heldout.get("mean_real_branch_fraction", 0.0),
            "heldout_mean_slot2_live_fraction": heldout.get("mean_slot2_live_fraction", 0.0),
            "heldout_mean_meso_branch_effect": heldout.get("mean_meso_branch_effect", 0.0),
            "heldout_mean_relation_handoff_drive": heldout.get("mean_relation_handoff_drive", 0.0),
            "heldout_mean_relation_attn_10": heldout.get("mean_relation_attn_10", 0.0),
            "heldout_mean_branch_positive_mask": heldout.get("mean_branch_positive_mask", 0.0),
            "heldout_mean_branch_negative_mask": heldout.get("mean_branch_negative_mask", 0.0),
            "heldout_mean_silent_singlepath_fraction": heldout.get("mean_silent_singlepath_fraction", 0.0),
            "heldout_branch_score": _branch_score(heldout),
            "nested_overall_read": nested.get("overall_read"),
            "nested_verdict_counts": nested.get("verdict_counts", {}),
            "law_token_aggregate_num_families": law_tokens["aggregate"]["num_law_families"],
            "law_token_aggregate_entropy": law_tokens["aggregate"]["law_family_entropy"],
            "law_token_aggregate_top_q_unique": law_tokens["aggregate"]["top_q_unique_count"],
            "continuity_mean_loop_autocorr_peak": continuity_circleworld["mean_loop_autocorr_peak"],
            "continuity_mean_nonlocal_chunk_repeat": continuity_circleworld["mean_nonlocal_chunk_repeat"],
            "overall_score": _overall_score(benchmark, heldout, nested),
        }
        rows.append(row)

    rows.sort(key=lambda item: item["overall_score"], reverse=True)
    summary = {
        "runtime": "circleworld_proto",
        "mode": "branchlaw_ablation_series",
        "base_config_path": str(base_config_path),
        "expanded_cases_path": str(expanded_cases_path),
        "device": device_name,
        "iterations": iterations,
        "population": population,
        "elite_count": elite_count,
        "clip_seconds": clip_seconds,
        "phase_blend": phase_blend,
        "rows": rows,
    }
    _write_json(out_dir / "branchlaw_ablation_summary.json", summary)

    lines = [
        "# Circleworld Branch-Law Ablation Series",
        "",
        f"- variants: {len(rows)}",
        f"- base config: `{base_config_path}`",
        "",
        "## Ranking",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['variant']}`: overall={row['overall_score']:.4f}, audio={row['benchmark_real_audio_score']:.4f}, "
            f"branch={row['heldout_branch_score']:.4f}, real_branch={row['heldout_mean_real_branch_fraction']:.4f}, "
            f"handoff={row['heldout_mean_relation_handoff_drive']:.4f}, nested={row['nested_overall_read']}"
        )
    (out_dir / "BRANCHLAW_ABLATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a Circleworld branch-law ablation series with shared sidecars.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    ap.add_argument("--expanded-cases", default=str(DEFAULT_EXPANDED_CASES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--population", type=int, default=6)
    ap.add_argument("--elite-count", type=int, default=2)
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    args = ap.parse_args()

    summary = run_series(
        out_dir=Path(args.out_dir),
        base_config_path=Path(args.base_config),
        expanded_cases_path=Path(args.expanded_cases),
        device_name=args.device,
        iterations=args.iterations,
        population=args.population,
        elite_count=args.elite_count,
        clip_seconds=args.clip_seconds,
        phase_blend=args.phase_blend,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
