from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


DEFAULT_INIT_CONFIG = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "training_run_2026-04-21_token_diverse_ramanujan"
    / "circleworld_real_anchor_config_cem_v1.json"
)
DEFAULT_CASES = ROOT / "outputs" / "circleworld_proto" / "expanded_anchor_cases_2026-04-09.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _branch_pressure_score_cfg() -> dict[str, float]:
    return {
        "target_corr": 0.77,
        "target_mae": 0.088,
        "target_dom": 0.60,
        "target_entropy": 0.57,
        "min_promotions": 8.0,
        "w_corr": 0.7,
        "w_mae": 2.2,
        "w_dom": 0.3,
        "w_entropy": 0.3,
        "w_promote": 0.05,
        "w_residue": 0.16,
        "w_promotability": 0.10,
        "w_major": 0.10,
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
        "w_real_branch": 3.0,
        "w_meso_branch": 55000.0,
        "w_slot2_live": 0.45,
        "w_silent_singlepath": 0.40,
        "w_branch_positive_mask": 3.0,
        "w_branch_negative_mask": 1.0,
        "w_decorative_slot2": 1.35,
        "w_relation_handoff": 4.5,
        "w_relation_attn_10": 0.75,
        "w_branch_defect": 1.2,
        "w_branch_world_grad": 0.8,
        "w_branch_q_disagreement": 0.8,
        "w_branch_phase_wall": 0.6,
        "w_branch_seed_energy": 0.25,
        "target_case_law_families": 2.75,
        "target_case_law_entropy": 0.28,
        "target_case_law_top_q_entropy": 0.12,
        "target_case_law_dominant_family_share": 0.78,
        "target_case_law_dominant_q_share": 0.60,
        "target_aggregate_law_families": 8.0,
        "target_aggregate_law_entropy": 0.52,
        "target_aggregate_law_top_q_entropy": 0.18,
        "target_aggregate_law_top_q_unique": 2.0,
        "target_aggregate_law_dominant_family_share": 0.60,
        "target_aggregate_law_dominant_q_share": 0.58,
        "w_case_law_family_shortfall": 0.6,
        "w_case_law_entropy_shortfall": 0.7,
        "w_case_law_top_q_entropy_shortfall": 0.5,
        "w_case_law_dominant_family": 0.5,
        "w_case_law_dominant_q": 0.4,
        "w_aggregate_law_family_shortfall": 0.9,
        "w_aggregate_law_entropy_shortfall": 1.1,
        "w_aggregate_law_top_q_entropy_shortfall": 0.9,
        "w_aggregate_law_top_q_unique_shortfall": 0.7,
        "w_aggregate_law_dominant_family": 1.2,
        "w_aggregate_law_dominant_q": 0.9,
    }


def _case_weights() -> dict[str, float]:
    return {
        "clarinet": 2.8,
        "metal-clang": 2.2,
        "mystic-chanting": 1.7,
        "anvil-impact": 1.5,
        "airplane": 1.3,
    }


def _extra_train_wavs() -> list[str]:
    return [
        str(ROOT / "wav_files" / "freewavesamples_ensoniq-zr-76-clarinet-c5_19f8dd5ddd.wav"),
    ]


def _extra_val_wavs() -> list[str]:
    return [
        str(ROOT / "wav_files" / "soundbible_mystic-chanting-4_7962fb220f.wav"),
        str(ROOT / "wav_files" / "soundbible_spooky-drone_7cc45e3229.wav"),
        str(ROOT / "wav_files" / "soundbible_muscle-car_2f21ca885f.wav"),
    ]


def _run_nested_commitment(config_path: Path, out_dir: Path, device_name: str, case_json: Path) -> dict[str, Any]:
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
        "native_multimode",
        "--case-json",
        str(case_json),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return json.loads((out_dir / "nested_commitment_report.json").read_text(encoding="utf-8"))


def run_experiment(
    out_dir: Path,
    checkpoint_dir: Path,
    init_config_path: Path,
    cases_path: Path,
    device_name: str,
    iterations: int,
    population: int,
    elite_count: int,
    seed: int,
    clip_seconds: int,
    phase_blend: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_summary = train_circleworld_real_anchor(
        out_dir=out_dir,
        checkpoint_dir=checkpoint_dir,
        iterations=iterations,
        population=population,
        elite_count=elite_count,
        seed=seed,
        device_name=device_name,
        clip_seconds=clip_seconds,
        phase_blend=phase_blend,
        init_config_path=init_config_path,
        score_cfg=_branch_pressure_score_cfg(),
        case_weights=_case_weights(),
        extra_train_wavs=_extra_train_wavs(),
        extra_val_wavs=_extra_val_wavs(),
    )

    config_path = Path(train_summary["checkpoint"])
    heldout = evaluate_config(
        config_path=config_path,
        out_dir=out_dir / "heldout_eval",
        time_steps=128,
        device_name=device_name,
    )
    benchmark = run_benchmark(
        config_path=config_path,
        out_dir=out_dir / "benchmark_expanded",
        device_name=device_name,
        clip_seconds=clip_seconds,
        phase_blend=phase_blend,
        rerender_check=True,
        cases=_load_cases_json(cases_path),
    )
    continuity_circleworld = benchmark_folder(
        folder=out_dir / "benchmark_expanded",
        pattern="*_circleworld.wav",
        out_path=out_dir / "benchmark_expanded" / "continuity_circleworld.json",
        min_loop_seconds=0.5,
        max_loop_seconds=4.0,
        chunk_seconds=2.0,
    )
    continuity_reference = benchmark_folder(
        folder=out_dir / "benchmark_expanded",
        pattern="*_reference.wav",
        out_path=out_dir / "benchmark_expanded" / "continuity_reference.json",
        min_loop_seconds=0.5,
        max_loop_seconds=4.0,
        chunk_seconds=2.0,
    )
    nested = _run_nested_commitment(
        config_path=config_path,
        out_dir=out_dir / "nested_commitment",
        device_name=device_name,
        case_json=cases_path,
    )
    law_tokens = build_library(
        config_path=config_path,
        out_dir=out_dir / "law_token_library",
        cases_path=cases_path,
        device_name=device_name,
        clip_seconds=clip_seconds,
    )

    summary = {
        "runtime": "circleworld_proto",
        "mode": "branch_pressure_experiment",
        "init_config_path": str(init_config_path),
        "cases_path": str(cases_path),
        "device": device_name,
        "iterations": iterations,
        "population": population,
        "elite_count": elite_count,
        "seed": seed,
        "clip_seconds": clip_seconds,
        "phase_blend": phase_blend,
        "checkpoint": str(config_path),
        "train_summary": str(out_dir / "train_summary.json"),
        "heldout_summary": str(out_dir / "heldout_eval" / "heldout_summary.json"),
        "benchmark_summary": str(out_dir / "benchmark_expanded" / "benchmark_summary.json"),
        "continuity_circleworld": str(out_dir / "benchmark_expanded" / "continuity_circleworld.json"),
        "continuity_reference": str(out_dir / "benchmark_expanded" / "continuity_reference.json"),
        "nested_summary": str(out_dir / "nested_commitment" / "nested_commitment_report.json"),
        "law_token_library": str(out_dir / "law_token_library" / "law_token_library.json"),
        "heldout": {
            "mean_slot2_live_fraction": heldout.get("mean_slot2_live_fraction", 0.0),
            "mean_real_branch_fraction": heldout.get("mean_real_branch_fraction", 0.0),
            "mean_meso_branch_effect": heldout.get("mean_meso_branch_effect", 0.0),
            "mean_branch_defect": heldout.get("mean_branch_defect", 0.0),
            "mean_branch_world_grad": heldout.get("mean_branch_world_grad", 0.0),
            "mean_branch_q_disagreement": heldout.get("mean_branch_q_disagreement", 0.0),
            "mean_branch_phase_wall": heldout.get("mean_branch_phase_wall", 0.0),
            "mean_branch_seed_energy": heldout.get("mean_branch_seed_energy", 0.0),
            "mean_num_law_families": heldout.get("mean_num_law_families", 0.0),
            "mean_dominant_law_family_share": heldout.get("mean_dominant_law_family_share", 0.0),
            "mean_law_family_entropy": heldout.get("mean_law_family_entropy", 0.0),
            "mean_dominant_law_q_share": heldout.get("mean_dominant_law_q_share", 0.0),
        },
        "benchmark": {
            "mean_corr": benchmark["mean_corr"],
            "mean_mae": benchmark["mean_mae"],
            "mean_mse": benchmark["mean_mse"],
        },
        "continuity": {
            "mean_loop_autocorr_peak": continuity_circleworld["mean_loop_autocorr_peak"],
            "mean_nonlocal_chunk_repeat": continuity_circleworld["mean_nonlocal_chunk_repeat"],
            "mean_adjacent_chunk_similarity": continuity_circleworld["mean_adjacent_chunk_similarity"],
        },
        "nested": {
            "overall_read": nested.get("overall_read"),
            "verdict_counts": nested.get("verdict_counts", {}),
        },
        "law_tokens": {
            "mean_num_law_packets": law_tokens["mean_num_law_packets"],
            "mean_num_law_families": law_tokens["mean_num_law_families"],
            "aggregate_num_families": law_tokens["aggregate_library"]["num_families"],
        },
    }
    _write_json(out_dir / "branch_pressure_experiment_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Circleworld branch-pressure training experiment with full sidecars.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--init-config", default=str(DEFAULT_INIT_CONFIG))
    ap.add_argument("--cases-json", default=str(DEFAULT_CASES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--elite-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260421)
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    args = ap.parse_args()

    summary = run_experiment(
        out_dir=Path(args.out_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        init_config_path=Path(args.init_config),
        cases_path=Path(args.cases_json),
        device_name=args.device,
        iterations=args.iterations,
        population=args.population,
        elite_count=args.elite_count,
        seed=args.seed,
        clip_seconds=args.clip_seconds,
        phase_blend=args.phase_blend,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
