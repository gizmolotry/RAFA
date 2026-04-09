from __future__ import annotations

import argparse
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import load_config
from circleworld import recurse_circleworld, summarize_circleworld_run
from export_circleworld_audio import _blend_phase, _load_circle_cfg, prepare_reference_audio
from stft_utils import compute_stft, inverse_stft
from train_circleworld import _candidate_from_mean_std, _graduation_anchor_wavs, _materialize_cfg, _safe_device, _set_seed


def _corrcoef_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.std(unbiased=False) * b.std(unbiased=False)
    if float(denom.item()) < 1e-12:
        return 0.0
    return float(((a * b).mean() / denom).item())


def _window_penalty(value: float, low: float | None, high: float | None) -> float:
    if low is not None and value < low:
        return float(low - value)
    if high is not None and value > high:
        return float(value - high)
    return 0.0


def _anchor_dataset(device_name: str, clip_seconds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = load_config()
    stft_cfg = cfg["data"]["stft"]
    train_wavs, val_wavs = _graduation_anchor_wavs()
    device = _safe_device(device_name)

    def _build(paths: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for wav_str in paths:
            wav_path = Path(wav_str)
            wav, _sr = prepare_reference_audio(
                wav_path=wav_path,
                device_name=device_name,
                clip_seconds_override=clip_seconds,
            )
            mag, phase = compute_stft(wav, stft_cfg)
            rows.append(
                {
                    "name": wav_path.stem,
                    "wav_path": str(wav_path),
                    "ref_wav": wav.squeeze(0).detach().to(device),
                    "mag": mag.detach().to(device),
                    "phase": phase.detach().to(device),
                    "phase_state": torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1).detach().to(device),
                }
            )
        return rows

    return _build(train_wavs), _build(val_wavs)


def _evaluate_cfg_on_dataset(
    circle_cfg,
    dataset: list[dict[str, Any]],
    phase_blend: float,
    score_cfg: dict[str, float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in dataset:
        run = recurse_circleworld(
            item["phase_state"],
            cfg=circle_cfg,
            depth=circle_cfg.recursion_depth,
            mode="active_packets",
        )
        summary = summarize_circleworld_run(run)
        final_phase = _blend_phase(item["phase"], run["phase_state"], blend=phase_blend)
        out_wav = inverse_stft(item["mag"], final_phase, load_config()["data"]["stft"]).squeeze(0)
        ref_wav = item["ref_wav"][: out_wav.numel()]
        out_wav = out_wav[: ref_wav.numel()]

        mse = float(torch.mean((ref_wav - out_wav) ** 2).item())
        mae = float(torch.mean(torch.abs(ref_wav - out_wav)).item())
        corr = _corrcoef_1d(ref_wav, out_wav)

        dom = float(summary["dominant_q_share"])
        ent = float(summary["q_entropy"])
        promos = float(summary["num_promotions"])
        corr_band = _window_penalty(
            corr,
            score_cfg.get("corr_low"),
            score_cfg.get("corr_high"),
        )
        mae_band = _window_penalty(
            mae,
            score_cfg.get("mae_low"),
            score_cfg.get("mae_high"),
        )
        score = (
            - score_cfg["w_corr"] * abs(corr - score_cfg["target_corr"])
            - score_cfg["w_mae"] * abs(mae - score_cfg["target_mae"])
            - score_cfg.get("w_corr_band", 0.0) * corr_band
            - score_cfg.get("w_mae_band", 0.0) * mae_band
            - score_cfg["w_dom"] * abs(dom - score_cfg["target_dom"])
            - score_cfg["w_entropy"] * abs(ent - score_cfg["target_entropy"])
            - score_cfg["w_promote"] * max(0.0, score_cfg["min_promotions"] - promos)
            + score_cfg["w_residue"] * float(summary["residue_drop"])
            + score_cfg["w_promotability"] * float(summary["promotability_gain"])
            + score_cfg["w_major"] * float(summary["major_gain"])
        )

        rows.append(
            {
                "name": item["name"],
                "wav_path": item["wav_path"],
                "score": float(score),
                "mse": mse,
                "mae": mae,
                "corr": corr,
                "corr_band_penalty": corr_band,
                "mae_band_penalty": mae_band,
                **summary,
            }
        )

    def _mean(key: str) -> float:
        return float(sum(r[key] for r in rows) / len(rows)) if rows else 0.0

    return {
        "num_samples": len(rows),
        "mean_score": _mean("score"),
        "mean_mse": _mean("mse"),
        "mean_mae": _mean("mae"),
        "mean_corr": _mean("corr"),
        "mean_corr_band_penalty": _mean("corr_band_penalty"),
        "mean_mae_band_penalty": _mean("mae_band_penalty"),
        "mean_major_gain": _mean("major_gain"),
        "mean_residue_drop": _mean("residue_drop"),
        "mean_promotability_gain": _mean("promotability_gain"),
        "mean_dominant_q_share": _mean("dominant_q_share"),
        "mean_q_entropy": _mean("q_entropy"),
        "mean_num_promotions": _mean("num_promotions"),
        "rows": rows,
    }


def train_circleworld_real_anchor(
    out_dir: Path,
    checkpoint_dir: Path,
    iterations: int,
    population: int,
    elite_count: int,
    seed: int,
    device_name: str,
    clip_seconds: int,
    phase_blend: float,
    init_config_path: Path,
    score_cfg: dict[str, float],
) -> dict[str, Any]:
    _set_seed(seed)
    device = _safe_device(device_name)
    base_cfg = _load_circle_cfg(init_config_path)
    train_set, val_set = _anchor_dataset(device_name=device_name, clip_seconds=clip_seconds)

    baseline_train = _evaluate_cfg_on_dataset(base_cfg, train_set, phase_blend=phase_blend, score_cfg=score_cfg)
    baseline_val = _evaluate_cfg_on_dataset(base_cfg, val_set, phase_blend=phase_blend, score_cfg=score_cfg)

    mean = {
        "q_weights": list(base_cfg.q_weights),
        "promotion_threshold": float(base_cfg.promotion_threshold),
        "max_promotions": float(base_cfg.max_promotions),
        "child_law_gain": float(base_cfg.child_law_gain),
        "attack_window": float(base_cfg.attack_window),
        "persistence_momentum": float(base_cfg.persistence_momentum),
    }
    std = {
        "q_weights": [0.15 for _ in base_cfg.q_weights],
        "promotion_threshold": 0.08,
        "max_promotions": 1.0,
        "child_law_gain": 0.10,
        "attack_window": 1.5,
        "persistence_momentum": 0.05,
    }

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    rng = random.Random(seed)

    for step in range(iterations):
        candidates: list[dict[str, Any]] = []
        for _ in range(population):
            cand = _candidate_from_mean_std(mean, std, rng)
            cfg = _materialize_cfg(base_cfg, cand)
            train_eval = _evaluate_cfg_on_dataset(cfg, train_set, phase_blend=phase_blend, score_cfg=score_cfg)
            val_eval = _evaluate_cfg_on_dataset(cfg, val_set, phase_blend=phase_blend, score_cfg=score_cfg)
            row = {
                "iteration": step,
                "candidate": cand,
                "materialized_cfg": {
                    "q_weights": list(cfg.q_weights),
                    "promotion_threshold": cfg.promotion_threshold,
                    "max_promotions": cfg.max_promotions,
                    "child_law_gain": cfg.child_law_gain,
                    "attack_window": cfg.attack_window,
                    "persistence_momentum": cfg.persistence_momentum,
                    "recursion_depth": cfg.recursion_depth,
                },
                "train": train_eval,
                "val": val_eval,
            }
            candidates.append(row)

        candidates.sort(key=lambda item: (item["val"]["mean_score"], item["train"]["mean_score"]), reverse=True)
        elites = candidates[:elite_count]
        history.extend(candidates)

        mean = {
            "q_weights": [
                sum(row["materialized_cfg"]["q_weights"][i] for row in elites) / len(elites)
                for i in range(len(base_cfg.q_weights))
            ],
            "promotion_threshold": sum(row["materialized_cfg"]["promotion_threshold"] for row in elites) / len(elites),
            "max_promotions": sum(row["materialized_cfg"]["max_promotions"] for row in elites) / len(elites),
            "child_law_gain": sum(row["materialized_cfg"]["child_law_gain"] for row in elites) / len(elites),
            "attack_window": sum(row["materialized_cfg"]["attack_window"] for row in elites) / len(elites),
            "persistence_momentum": sum(row["materialized_cfg"]["persistence_momentum"] for row in elites) / len(elites),
        }

        def _elite_std(values: list[float], floor: float) -> float:
            if len(values) <= 1:
                return floor
            mean_v = sum(values) / len(values)
            var = sum((v - mean_v) ** 2 for v in values) / len(values)
            return max(floor, math.sqrt(var) * 0.85)

        std = {
            "q_weights": [
                _elite_std([row["materialized_cfg"]["q_weights"][i] for row in elites], 0.03)
                for i in range(len(base_cfg.q_weights))
            ],
            "promotion_threshold": _elite_std([row["materialized_cfg"]["promotion_threshold"] for row in elites], 0.02),
            "max_promotions": _elite_std([float(row["materialized_cfg"]["max_promotions"]) for row in elites], 0.5),
            "child_law_gain": _elite_std([row["materialized_cfg"]["child_law_gain"] for row in elites], 0.02),
            "attack_window": _elite_std([float(row["materialized_cfg"]["attack_window"]) for row in elites], 0.5),
            "persistence_momentum": _elite_std([row["materialized_cfg"]["persistence_momentum"] for row in elites], 0.02),
        }

        candidate_best = deepcopy(candidates[0])
        if best_state is None or candidate_best["val"]["mean_score"] > best_state["val"]["mean_score"]:
            best_state = candidate_best

    assert best_state is not None
    best_cfg = _materialize_cfg(base_cfg, best_state["materialized_cfg"])
    best_train = _evaluate_cfg_on_dataset(best_cfg, train_set, phase_blend=phase_blend, score_cfg=score_cfg)
    best_val = _evaluate_cfg_on_dataset(best_cfg, val_set, phase_blend=phase_blend, score_cfg=score_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "circleworld_real_anchor_config_cem_v1.json"
    history_path = out_dir / "search_history.json"
    summary_path = out_dir / "train_summary.json"

    payload = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0_real_anchor",
        "trainer": "cem_real_anchor",
        "seed": seed,
        "device": str(device),
        "clip_seconds": clip_seconds,
        "phase_blend": phase_blend,
        "init_config_path": str(init_config_path),
        "score_cfg": score_cfg,
        "train_anchor_wavs": [x["wav_path"] for x in train_set],
        "val_anchor_wavs": [x["wav_path"] for x in val_set],
        "config": {
            "q_weights": list(best_cfg.q_weights),
            "promotion_threshold": best_cfg.promotion_threshold,
            "max_promotions": best_cfg.max_promotions,
            "child_law_gain": best_cfg.child_law_gain,
            "attack_window": best_cfg.attack_window,
            "persistence_momentum": best_cfg.persistence_momentum,
            "recursion_depth": best_cfg.recursion_depth,
        },
    }
    ckpt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0_real_anchor",
        "status": "completed",
        "trainer": "cem_real_anchor",
        "seed": seed,
        "device": str(device),
        "iterations": iterations,
        "population": population,
        "elite_count": elite_count,
        "clip_seconds": clip_seconds,
        "phase_blend": phase_blend,
        "checkpoint": str(ckpt_path),
        "baseline_train": baseline_train,
        "baseline_val": baseline_val,
        "best_train": best_train,
        "best_val": best_val,
        "best_candidate": best_state,
        "best_config": payload["config"],
        "artifact_paths": {
            "summary": str(summary_path),
            "search_history": str(history_path),
            "checkpoint": str(ckpt_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Circleworld directly on real-anchor audio comparisons.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--elite-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260409)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--init-config", required=True)
    ap.add_argument("--target-corr", type=float, default=0.78)
    ap.add_argument("--target-mae", type=float, default=0.08)
    ap.add_argument("--target-dom", type=float, default=0.62)
    ap.add_argument("--target-entropy", type=float, default=0.58)
    ap.add_argument("--min-promotions", type=float, default=8.0)
    ap.add_argument("--w-corr", type=float, default=1.0)
    ap.add_argument("--w-mae", type=float, default=3.0)
    ap.add_argument("--w-dom", type=float, default=0.4)
    ap.add_argument("--w-entropy", type=float, default=0.4)
    ap.add_argument("--w-promote", type=float, default=0.05)
    ap.add_argument("--w-residue", type=float, default=0.2)
    ap.add_argument("--w-promotability", type=float, default=0.1)
    ap.add_argument("--w-major", type=float, default=0.1)
    ap.add_argument("--corr-low", type=float, default=None)
    ap.add_argument("--corr-high", type=float, default=None)
    ap.add_argument("--mae-low", type=float, default=None)
    ap.add_argument("--mae-high", type=float, default=None)
    ap.add_argument("--w-corr-band", type=float, default=0.0)
    ap.add_argument("--w-mae-band", type=float, default=0.0)
    args = ap.parse_args()

    score_cfg = {
        "target_corr": float(args.target_corr),
        "target_mae": float(args.target_mae),
        "target_dom": float(args.target_dom),
        "target_entropy": float(args.target_entropy),
        "min_promotions": float(args.min_promotions),
        "w_corr": float(args.w_corr),
        "w_mae": float(args.w_mae),
        "w_dom": float(args.w_dom),
        "w_entropy": float(args.w_entropy),
        "w_promote": float(args.w_promote),
        "w_residue": float(args.w_residue),
        "w_promotability": float(args.w_promotability),
        "w_major": float(args.w_major),
        "corr_low": None if args.corr_low is None else float(args.corr_low),
        "corr_high": None if args.corr_high is None else float(args.corr_high),
        "mae_low": None if args.mae_low is None else float(args.mae_low),
        "mae_high": None if args.mae_high is None else float(args.mae_high),
        "w_corr_band": float(args.w_corr_band),
        "w_mae_band": float(args.w_mae_band),
    }

    summary = train_circleworld_real_anchor(
        out_dir=Path(args.out_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        iterations=args.iterations,
        population=args.population,
        elite_count=args.elite_count,
        seed=args.seed,
        device_name=args.device,
        clip_seconds=args.clip_seconds,
        phase_blend=args.phase_blend,
        init_config_path=Path(args.init_config),
        score_cfg=score_cfg,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
