from __future__ import annotations

import argparse
import json
import math
import random
import sys
import wave
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablate_formalization import _generate_seed_phase, _load_cfg
from circleworld import CircleworldConfig, circleworld_loss, recurse_circleworld, summarize_circleworld_run
from lib_blackwell import NakedRAFA
from config import load_config
from stft_utils import compute_stft
from rafa_math_tools import phase_to_phasor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _default_circle_cfg() -> CircleworldConfig:
    raw = _load_cfg().get("circleworld", {})
    return CircleworldConfig(
        qset=tuple(raw.get("qset", (2, 3, 4, 5, 6, 8, 12))),
        q_weights=tuple(raw.get("q_weights", (1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5))),
        promotion_threshold=float(raw.get("promotion_threshold", 0.58)),
        max_promotions=int(raw.get("max_promotions", 4)),
        recursion_depth=int(raw.get("recursion_depth", 2)),
        residue_scale=float(raw.get("residue_scale", 0.75)),
        child_law_gain=float(raw.get("child_law_gain", 0.25)),
        attack_window=int(raw.get("attack_window", 8)),
        persistence_momentum=float(raw.get("persistence_momentum", 0.6)),
    )


def _load_circle_cfg(path: Path) -> CircleworldConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["config"] if "config" in payload else payload
    return CircleworldConfig(
        qset=tuple(cfg.get("qset", (2, 3, 4, 5, 6, 8, 12))),
        q_weights=tuple(cfg["q_weights"]),
        promotion_threshold=float(cfg["promotion_threshold"]),
        max_promotions=int(cfg["max_promotions"]),
        recursion_depth=int(cfg["recursion_depth"]),
        residue_scale=float(cfg.get("residue_scale", 0.75)),
        child_law_gain=float(cfg["child_law_gain"]),
        attack_window=int(cfg["attack_window"]),
        persistence_momentum=float(cfg["persistence_momentum"]),
    )


def _materialize_cfg(base: CircleworldConfig, vec: dict[str, Any]) -> CircleworldConfig:
    return CircleworldConfig(
        qset=base.qset,
        q_weights=tuple(float(max(0.1, x)) for x in vec["q_weights"]),
        promotion_threshold=float(min(0.9, max(0.25, vec["promotion_threshold"]))),
        max_promotions=int(min(8, max(1, round(vec["max_promotions"])))),
        recursion_depth=base.recursion_depth,
        residue_scale=base.residue_scale,
        child_law_gain=float(min(1.0, max(0.01, vec["child_law_gain"]))),
        attack_window=int(min(24, max(2, round(vec["attack_window"])))),
        persistence_momentum=float(min(0.98, max(0.05, vec["persistence_momentum"]))),
    )


def _candidate_from_mean_std(mean: dict[str, Any], std: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    out = {
        "q_weights": [],
        "promotion_threshold": rng.gauss(mean["promotion_threshold"], std["promotion_threshold"]),
        "max_promotions": rng.gauss(mean["max_promotions"], std["max_promotions"]),
        "child_law_gain": rng.gauss(mean["child_law_gain"], std["child_law_gain"]),
        "attack_window": rng.gauss(mean["attack_window"], std["attack_window"]),
        "persistence_momentum": rng.gauss(mean["persistence_momentum"], std["persistence_momentum"]),
    }
    for idx, val in enumerate(mean["q_weights"]):
        out["q_weights"].append(rng.gauss(val, std["q_weights"][idx]))
    return out


def _seed_plan(
    device: torch.device,
    train_count: int,
    val_count: int,
    include_naked_rafa: bool = True,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    train: list[tuple[str, int]] = [("synthetic", 1100 + i) for i in range(train_count)]
    val: list[tuple[str, int]] = [("synthetic", 2100 + i) for i in range(val_count)]
    if device.type == "cuda" and include_naked_rafa:
        train.extend([("naked_rafa", 3100 + i) for i in range(max(2, train_count // 2))])
        val.extend([("naked_rafa", 4100 + i) for i in range(max(2, val_count // 2))])
    return train, val


def _graduation_anchor_wavs() -> tuple[list[str], list[str]]:
    train = [
        "D:\\RAFA\\wav_files\\soundbible_muscle-car_9fce13702c.wav",
        "D:\\RAFA\\wav_files\\soundbible_steam-engine-running_68e20d27d1.wav",
        "D:\\RAFA\\wav_files\\soundbible_mystic-chanting-4_a38f2bbe68.wav",
        "D:\\RAFA\\wav_files\\soundbible_metal-clang_d06ac0429e.wav",
        "D:\\RAFA\\wav_files\\soundbible_spooky-drone_2efbfd965b.wav",
        "D:\\RAFA\\wav_files\\soundbible_airplane-takeoff_c752b8ba8b.wav",
    ]
    val = [
        "D:\\RAFA\\wav_files\\soundbible_muscle-car_2f21ca885f.wav",
        "D:\\RAFA\\wav_files\\soundbible_mystic-chanting-4_7962fb220f.wav",
        "D:\\RAFA\\wav_files\\soundbible_anvil-impact_c8021375ce.wav",
        "D:\\RAFA\\wav_files\\soundbible_spooky-drone_7cc45e3229.wav",
        "D:\\RAFA\\wav_files\\soundbible_airplane-landing-airport_caf74bfdeb.wav",
    ]
    return train, val


def _load_local_pcm_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        nframes = int(wf.getnframes())
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = 1 << 23
        v = (v ^ sign) - sign
        arr = v.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    arr = arr.reshape(-1, ch).T
    wav = torch.from_numpy(arr)
    return wav, sr


def _native_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    ratio = float(target_sr) / float(orig_sr)
    target_len = int(round(wav.size(1) * ratio))
    return F.interpolate(
        wav.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def _crop_or_tile(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.size(1) < target_len:
        reps = int(np.ceil(float(target_len) / float(max(1, wav.size(1)))))
        wav = wav.repeat(1, reps)
    return wav[:, :target_len]


def _phase_state_from_wav(
    wav_path: Path,
    target_time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    cfg = load_config()
    sr = int(cfg["data"]["sample_rate"])
    stft_cfg = cfg["data"]["stft"]
    hop = int(stft_cfg["hop"])
    winl = int(stft_cfg["win_length"])
    target_len = hop * max(1, target_time_steps - 1) + winl

    wav, wav_sr = _load_local_pcm_wav(wav_path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = _native_resample(wav, wav_sr, sr)
    wav = _crop_or_tile(wav, target_len).to(device)
    mag, phase = compute_stft(wav, stft_cfg)
    if phase.size(-1) > target_time_steps:
        phase = phase[..., :target_time_steps]
    elif phase.size(-1) < target_time_steps:
        pad = target_time_steps - phase.size(-1)
        phase = F.pad(phase, (0, pad), mode="replicate")
    return phase_to_phasor(phase).detach()


def _build_dataset(
    plan: list[tuple[str, int]],
    time_steps: int,
    device: torch.device,
    anchor_wavs: list[str] | None = None,
) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    rafa_core: NakedRAFA | None = NakedRAFA(dev=device.type) if any(src == "naked_rafa" for src, _ in plan) else None
    for source, seed in plan:
        _set_seed(seed)
        phase_state = _generate_seed_phase(
            rafa_core=rafa_core,
            batch_size=1,
            time_steps=time_steps,
            device=device,
            seed_source=source,
        ).detach()
        dataset.append(
            {
                "source": source,
                "seed": seed,
                "phase_state": phase_state,
            }
        )
    for wav_path in anchor_wavs or []:
        phase_state = _phase_state_from_wav(Path(wav_path), target_time_steps=time_steps, device=device)
        dataset.append(
            {
                "source": "graduation_anchor",
                "seed": int(abs(hash(wav_path)) % 10_000_000),
                "wav_path": str(wav_path),
                "phase_state": phase_state,
            }
        )
    return dataset


def _score_run(
    run: dict[str, Any],
    score_cfg: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    summary = summarize_circleworld_run(run)
    losses = circleworld_loss(run)
    score_cfg = score_cfg or {}
    target_dom = float(score_cfg.get("target_dom", 0.70))
    target_entropy = float(score_cfg.get("target_entropy", 0.45))
    min_promotions = float(score_cfg.get("min_promotions", 3.0))
    w_major = float(score_cfg.get("w_major", 1.0))
    w_residue = float(score_cfg.get("w_residue", 1.0))
    w_promo = float(score_cfg.get("w_promo", 0.5))
    w_final_major = float(score_cfg.get("w_final_major", 0.25))
    w_final_minor = float(score_cfg.get("w_final_minor", 0.25))
    w_dom = float(score_cfg.get("w_dom", 0.40))
    w_entropy = float(score_cfg.get("w_entropy", 0.20))
    w_major_sat = float(score_cfg.get("w_major_sat", 0.50))
    w_loss = float(score_cfg.get("w_loss", 0.05))
    w_dom_target = float(score_cfg.get("w_dom_target", 0.0))
    w_entropy_target = float(score_cfg.get("w_entropy_target", 0.0))
    w_promote_floor = float(score_cfg.get("w_promote_floor", 0.0))
    dom_target_err = abs(summary["dominant_q_share"] - target_dom)
    entropy_target_err = abs(summary["q_entropy"] - target_entropy)
    promote_floor = max(0.0, min_promotions - summary["num_promotions"])
    score = (
        w_major * summary["major_gain"]
        + w_residue * summary["residue_drop"]
        + w_promo * summary["promotability_gain"]
        + w_final_major * summary["final_major_mass"]
        - w_final_minor * summary["final_minor_residue"]
        - w_dom * summary["dominant_q_share"]
        + w_entropy * summary["q_entropy"]
        - w_major_sat * summary["major_saturation"]
        - w_loss * float(losses["loss"].item())
        - w_dom_target * dom_target_err
        - w_entropy_target * entropy_target_err
        - w_promote_floor * promote_floor
    )
    metrics = {
        **summary,
        "loss": float(losses["loss"].item()),
        "l_q_dom": float(losses["l_q_dom"].item()),
        "l_q_entropy": float(losses["l_q_entropy"].item()),
        "l_major_sat": float(losses["l_major_sat"].item()),
        "dom_target_err": float(dom_target_err),
        "entropy_target_err": float(entropy_target_err),
        "promote_floor_penalty": float(promote_floor),
        "score": float(score),
    }
    return float(score), metrics


def _evaluate_cfg(
    cfg: CircleworldConfig,
    dataset: list[dict[str, Any]],
    mode: str,
    score_cfg: dict[str, float] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in dataset:
        run = recurse_circleworld(item["phase_state"], cfg=cfg, depth=cfg.recursion_depth, mode=mode)
        score, metrics = _score_run(run, score_cfg=score_cfg)
        rows.append(
            {
                "source": item["source"],
                "seed": int(item["seed"]),
                "score": score,
                **metrics,
            }
        )

    def _mean(key: str) -> float:
        return float(sum(row[key] for row in rows) / max(1, len(rows)))

    return {
        "num_samples": len(rows),
        "mean_score": _mean("score"),
        "mean_loss": _mean("loss"),
        "mean_major_gain": _mean("major_gain"),
        "mean_residue_drop": _mean("residue_drop"),
        "mean_promotability_gain": _mean("promotability_gain"),
        "mean_final_major_mass": _mean("final_major_mass"),
        "mean_final_minor_residue": _mean("final_minor_residue"),
        "mean_dominant_q_share": _mean("dominant_q_share"),
        "mean_q_entropy": _mean("q_entropy"),
        "mean_major_saturation": _mean("major_saturation"),
        "mean_l_q_dom": _mean("l_q_dom"),
        "mean_l_q_entropy": _mean("l_q_entropy"),
        "mean_l_major_sat": _mean("l_major_sat"),
        "mean_dom_target_err": _mean("dom_target_err"),
        "mean_entropy_target_err": _mean("entropy_target_err"),
        "mean_promote_floor_penalty": _mean("promote_floor_penalty"),
        "mean_num_promotions": _mean("num_promotions"),
        "rows": rows,
    }


def _baseline_report(
    base_cfg: CircleworldConfig,
    train_set: list[dict[str, Any]],
    val_set: list[dict[str, Any]],
    score_cfg: dict[str, float] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for mode in ("no_promotion", "passive_packets", "active_packets"):
        report[mode] = {
            "train": _evaluate_cfg(base_cfg, train_set, mode=mode, score_cfg=score_cfg),
            "val": _evaluate_cfg(base_cfg, val_set, mode=mode, score_cfg=score_cfg),
        }
    return report


def train_circleworld(
    out_dir: Path,
    checkpoint_dir: Path,
    iterations: int,
    population: int,
    elite_count: int,
    time_steps: int,
    recursion_depth: int,
    device_name: str,
    train_count: int,
    val_count: int,
    seed: int,
    init_config_path: Path | None = None,
    score_cfg: dict[str, float] | None = None,
    include_graduation_anchors: bool = False,
    include_naked_rafa: bool = True,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    _set_seed(seed)
    base_cfg = _load_circle_cfg(init_config_path) if init_config_path else _default_circle_cfg()
    base_cfg.recursion_depth = recursion_depth

    train_plan, val_plan = _seed_plan(
        device,
        train_count=train_count,
        val_count=val_count,
        include_naked_rafa=include_naked_rafa,
    )
    grad_train_wavs, grad_val_wavs = _graduation_anchor_wavs() if include_graduation_anchors else ([], [])
    train_set = _build_dataset(train_plan, time_steps=time_steps, device=device, anchor_wavs=grad_train_wavs)
    val_set = _build_dataset(val_plan, time_steps=time_steps, device=device, anchor_wavs=grad_val_wavs)

    baseline = _baseline_report(base_cfg, train_set, val_set, score_cfg=score_cfg)

    mean = {
        "q_weights": list(base_cfg.q_weights),
        "promotion_threshold": float(base_cfg.promotion_threshold),
        "max_promotions": float(base_cfg.max_promotions),
        "child_law_gain": float(base_cfg.child_law_gain),
        "attack_window": float(base_cfg.attack_window),
        "persistence_momentum": float(base_cfg.persistence_momentum),
    }
    std = {
        "q_weights": [0.20 for _ in base_cfg.q_weights],
        "promotion_threshold": 0.10,
        "max_promotions": 1.25,
        "child_law_gain": 0.12,
        "attack_window": 2.0,
        "persistence_momentum": 0.10,
    }

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    rng = random.Random(seed)

    for step in range(iterations):
        candidates: list[dict[str, Any]] = []
        for _ in range(population):
            cand = _candidate_from_mean_std(mean, std, rng)
            cfg = _materialize_cfg(base_cfg, cand)
            train_eval = _evaluate_cfg(cfg, train_set, mode="active_packets", score_cfg=score_cfg)
            val_eval = _evaluate_cfg(cfg, val_set, mode="active_packets", score_cfg=score_cfg)
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
                _elite_std([row["materialized_cfg"]["q_weights"][i] for row in elites], 0.04)
                for i in range(len(base_cfg.q_weights))
            ],
            "promotion_threshold": _elite_std([row["materialized_cfg"]["promotion_threshold"] for row in elites], 0.03),
            "max_promotions": _elite_std([float(row["materialized_cfg"]["max_promotions"]) for row in elites], 0.50),
            "child_law_gain": _elite_std([row["materialized_cfg"]["child_law_gain"] for row in elites], 0.03),
            "attack_window": _elite_std([float(row["materialized_cfg"]["attack_window"]) for row in elites], 0.75),
            "persistence_momentum": _elite_std([row["materialized_cfg"]["persistence_momentum"] for row in elites], 0.03),
        }

        candidate_best = deepcopy(candidates[0])
        if best_state is None or candidate_best["val"]["mean_score"] > best_state["val"]["mean_score"]:
            best_state = candidate_best

    assert best_state is not None
    best_cfg = _materialize_cfg(base_cfg, best_state["materialized_cfg"])
    best_train = _evaluate_cfg(best_cfg, train_set, mode="active_packets", score_cfg=score_cfg)
    best_val = _evaluate_cfg(best_cfg, val_set, mode="active_packets", score_cfg=score_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_cfg_path = checkpoint_dir / "circleworld_config_cem_v1.json"
    history_path = out_dir / "search_history.json"
    summary_path = out_dir / "train_summary.json"
    baseline_path = out_dir / "baseline_report.json"

    best_config_payload = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0",
        "trainer": "cem_parameter_search",
        "mode": "active_packets",
        "seed": seed,
        "device": str(device),
        "time_steps": time_steps,
        "init_config_path": str(init_config_path) if init_config_path else None,
        "include_graduation_anchors": include_graduation_anchors,
        "include_naked_rafa": include_naked_rafa,
        "score_cfg": score_cfg or {},
        "train_plan": [{"source": src, "seed": s} for src, s in train_plan],
        "val_plan": [{"source": src, "seed": s} for src, s in val_plan],
        "graduation_anchor_train_wavs": grad_train_wavs,
        "graduation_anchor_val_wavs": grad_val_wavs,
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
    best_cfg_path.write_text(json.dumps(best_config_payload, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0",
        "status": "completed",
        "trainer": "cem_parameter_search",
        "seed": seed,
        "device": str(device),
        "iterations": iterations,
        "population": population,
        "elite_count": elite_count,
        "time_steps": time_steps,
        "checkpoint": str(best_cfg_path),
        "baseline_active_val": baseline["active_packets"]["val"],
        "best_active_train": best_train,
        "best_active_val": best_val,
        "best_candidate": best_state,
        "best_config": best_config_payload["config"],
        "artifact_paths": {
            "summary": str(summary_path),
            "baseline_report": str(baseline_path),
            "search_history": str(history_path),
            "checkpoint": str(best_cfg_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Circleworld parameters in an isolated prototype lane.")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "circleworld_proto" / "proper_attempt_2026-04-06"))
    ap.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints_circleworld_proto" / "proper_attempt_2026-04-06"))
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--elite-count", type=int, default=3)
    ap.add_argument("--time-steps", type=int, default=96)
    ap.add_argument("--recursion-depth", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-count", type=int, default=6)
    ap.add_argument("--val-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--init-config", default=None)
    ap.add_argument("--target-dom", type=float, default=0.70)
    ap.add_argument("--target-entropy", type=float, default=0.45)
    ap.add_argument("--min-promotions", type=float, default=3.0)
    ap.add_argument("--w-dom-target", type=float, default=0.0)
    ap.add_argument("--w-entropy-target", type=float, default=0.0)
    ap.add_argument("--w-promote-floor", type=float, default=0.0)
    ap.add_argument("--include-graduation-anchors", action="store_true")
    ap.add_argument("--no-naked-rafa", action="store_true")
    args = ap.parse_args()

    score_cfg = {
        "target_dom": float(args.target_dom),
        "target_entropy": float(args.target_entropy),
        "min_promotions": float(args.min_promotions),
        "w_dom_target": float(args.w_dom_target),
        "w_entropy_target": float(args.w_entropy_target),
        "w_promote_floor": float(args.w_promote_floor),
    }

    summary = train_circleworld(
        out_dir=Path(args.out_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        iterations=args.iterations,
        population=args.population,
        elite_count=args.elite_count,
        time_steps=args.time_steps,
        recursion_depth=args.recursion_depth,
        device_name=args.device,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
        init_config_path=Path(args.init_config) if args.init_config else None,
        score_cfg=score_cfg,
        include_graduation_anchors=bool(args.include_graduation_anchors),
        include_naked_rafa=not bool(args.no_naked_rafa),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
