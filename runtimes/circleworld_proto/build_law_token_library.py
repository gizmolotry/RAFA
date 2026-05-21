from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from circleworld import (
    build_law_token_library,
    recurse_circleworld,
    serialize_law_token_library,
    summarize_circleworld_run,
)
from export_circleworld_audio import _load_circle_cfg, prepare_reference_audio
from rafa_relational_signature import (
    build_relational_signature_library,
    concat_relational_signature_banks,
    serialize_relational_signature_bank,
    serialize_relational_signature_library,
)
from rafa_math_tools import phase_to_phasor
from stft_utils import compute_stft
from config import load_config


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _serialize_law_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_index": int(packet["batch_index"].item()),
        "time_index": int(packet["time_index"].item()),
        "depth_index": int(packet["depth_index"].item()),
        "score": float(packet["score"].item()),
        "major_mass": float(packet["major_mass"].item()),
        "minor_residue": float(packet["minor_residue"].item()),
        "harmonic_ratio": float(packet["harmonic_ratio"].item()),
        "persistence": float(packet["persistence"].item()),
        "promotability": float(packet["promotability"].item()),
        "concentration": float(packet["concentration"].item()),
        "sharpness": float(packet["sharpness"].item()),
        "pair_strength": float(packet["pair_strength"].item()),
        "q_mass": packet["q_mass"].detach().cpu().view(-1).tolist(),
        "law_signature": packet["law_signature"].detach().cpu().view(-1).tolist(),
        "law_window": [int(x) for x in packet["law_window"].detach().cpu().view(-1).tolist()],
    }


def _default_cases_path() -> Path:
    return ROOT / "outputs" / "circleworld_proto" / "expanded_anchor_cases_2026-04-09.json"


def _load_cases(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cases file must be a JSON object: {path}")
    return {str(k): str(v) for k, v in payload.items()}


def _family_count_metrics(library: dict[str, Any]) -> dict[str, float]:
    counts = [float(family.get("count", 0.0)) for family in library.get("families", [])]
    total = sum(counts)
    if total <= 0.0:
        return {
            "effective_family_count": 0.0,
            "family_entropy": 0.0,
            "dominant_family_pressure": 0.0,
        }
    probs = [count / total for count in counts if count > 0.0]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    effective = math.exp(entropy)
    dominant = max(probs)
    return {
        "effective_family_count": float(effective),
        "family_entropy": float(entropy),
        "dominant_family_pressure": float(dominant),
    }


def _stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / float(len(values))
    return float((sum((value - mean) ** 2 for value in values) / float(len(values))) ** 0.5)


def build_library(
    config_path: Path,
    out_dir: Path,
    cases_path: Path,
    device_name: str,
    clip_seconds: int,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    stft_cfg = load_config()["data"]["stft"]
    circle_cfg = _load_circle_cfg(config_path)
    run_mode = circle_cfg.branching_mode if circle_cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "active_packets"
    cases = _load_cases(cases_path)

    case_rows: list[dict[str, Any]] = []
    all_law_packets: list[dict[str, Any]] = []
    all_signature_banks = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for case_name, wav_path_str in cases.items():
        wav_path = Path(wav_path_str)
        wav, _ = prepare_reference_audio(
            wav_path=wav_path,
            device_name=device_name,
            clip_seconds_override=clip_seconds,
        )
        wav = wav.to(device)
        mag, phase = compute_stft(wav, stft_cfg)
        phase_state = phase_to_phasor(phase)
        run = recurse_circleworld(phase_state, cfg=circle_cfg, depth=circle_cfg.recursion_depth, mode=run_mode)
        summary = summarize_circleworld_run(run)
        library = serialize_law_token_library(run["law_token_library"])
        rel_bank = serialize_relational_signature_bank(run["relational_signatures"])
        rel_library = serialize_relational_signature_library(run["relational_signature_library"])
        case_law_packets = [_serialize_law_packet(packet) for packet in run["law_packets"]]
        all_law_packets.extend(run["law_packets"])
        all_signature_banks.append(run["relational_signatures"])
        row = {
            "case": case_name,
            "wav_path": str(wav_path),
            "summary": summary,
            "law_token_library": library,
            "relational_signature_library": rel_library,
            "relational_signatures": rel_bank,
            "law_packets": case_law_packets,
        }
        case_rows.append(row)
        (out_dir / f"{case_name}_law_tokens.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        (out_dir / f"{case_name}_relational_signatures.json").write_text(json.dumps(rel_bank, indent=2), encoding="utf-8")

    aggregate_library = serialize_law_token_library(build_law_token_library(all_law_packets, circle_cfg))
    aggregate_signature_bank = concat_relational_signature_banks(all_signature_banks)
    aggregate_signature_library = serialize_relational_signature_library(
        build_relational_signature_library(aggregate_signature_bank, circle_cfg)
    )
    aggregate_law_family_metrics = _family_count_metrics(aggregate_library)
    aggregate_signature_family_metrics = _family_count_metrics(aggregate_signature_library)
    aggregate_packets = [_serialize_law_packet(packet) for packet in all_law_packets]
    ranking = sorted(
        [
            {
                "case": row["case"],
                "num_law_packets": row["summary"].get("num_law_packets", 0.0),
                "num_law_families": row["summary"].get("num_law_families", 0.0),
                "num_relational_signatures": row["summary"].get("num_relational_signatures", 0.0),
                "num_relational_signature_families": row["summary"].get("num_relational_signature_families", 0.0),
                "major_gain": row["summary"].get("major_gain", 0.0),
                "promotability_gain": row["summary"].get("promotability_gain", 0.0),
            }
            for row in case_rows
        ],
        key=lambda item: (
            float(item["num_relational_signatures"]),
            float(item["num_relational_signature_families"]),
            float(item["num_law_packets"]),
            float(item["num_law_families"]),
            float(item["promotability_gain"]),
        ),
        reverse=True,
    )

    summary = {
        "runtime": "circleworld_proto",
        "mode": "law_token_library",
        "config_path": str(config_path),
        "cases_path": str(cases_path),
        "device": str(device),
        "clip_seconds": int(clip_seconds),
        "run_mode": run_mode,
        "num_cases": len(case_rows),
        "mean_num_law_packets": float(sum(row["summary"].get("num_law_packets", 0.0) for row in case_rows) / max(1, len(case_rows))),
        "mean_num_law_families": float(sum(row["summary"].get("num_law_families", 0.0) for row in case_rows) / max(1, len(case_rows))),
        "mean_law_family_size": float(sum(row["summary"].get("mean_law_family_size", 0.0) for row in case_rows) / max(1, len(case_rows))),
        "mean_num_relational_signatures": float(sum(row["summary"].get("num_relational_signatures", 0.0) for row in case_rows) / max(1, len(case_rows))),
        "mean_num_relational_signature_families": float(sum(row["summary"].get("num_relational_signature_families", 0.0) for row in case_rows) / max(1, len(case_rows))),
        "mean_relational_signature_confidence": float(sum(row["summary"].get("mean_relational_signature_confidence", 0.0) for row in case_rows) / max(1, len(case_rows))),
        "mean_major_gain": float(sum(row["summary"].get("major_gain", 0.0) for row in case_rows) / max(1, len(case_rows))),
        "mean_promotability_gain": float(sum(row["summary"].get("promotability_gain", 0.0) for row in case_rows) / max(1, len(case_rows))),
        "aggregate_law_effective_family_count": aggregate_law_family_metrics["effective_family_count"],
        "aggregate_law_family_entropy": aggregate_law_family_metrics["family_entropy"],
        "aggregate_law_dominant_family_pressure": aggregate_law_family_metrics["dominant_family_pressure"],
        "aggregate_relational_effective_family_count": aggregate_signature_family_metrics["effective_family_count"],
        "aggregate_relational_family_entropy": aggregate_signature_family_metrics["family_entropy"],
        "aggregate_relational_dominant_family_pressure": aggregate_signature_family_metrics["dominant_family_pressure"],
        "case_law_family_count_std": _stddev([float(row["summary"].get("num_law_families", 0.0)) for row in case_rows]),
        "case_relational_signature_family_count_std": _stddev(
            [float(row["summary"].get("num_relational_signature_families", 0.0)) for row in case_rows]
        ),
        "case_relational_signature_confidence_std": _stddev(
            [float(row["summary"].get("mean_relational_signature_confidence", 0.0)) for row in case_rows]
        ),
        "case_relational_branch_mass_std": _stddev(
            [float(row["summary"].get("mean_relational_branch_mass", 0.0)) for row in case_rows]
        ),
        "aggregate_library": aggregate_library,
        "aggregate_relational_signature_library": aggregate_signature_library,
        "case_ranking": ranking,
        "cases": case_rows,
    }
    (out_dir / "law_token_library.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "law_packets.json").write_text(json.dumps(aggregate_packets, indent=2), encoding="utf-8")
    (out_dir / "relational_signature_library.json").write_text(json.dumps(aggregate_signature_library, indent=2), encoding="utf-8")
    (out_dir / "relational_signatures.json").write_text(json.dumps(serialize_relational_signature_bank(aggregate_signature_bank), indent=2), encoding="utf-8")

    md_lines = [
        "# Circleworld Law Token Library",
        "",
        f"- config: `{config_path}`",
        f"- cases: `{cases_path}`",
        f"- device: `{device}`",
        f"- run mode: `{run_mode}`",
        f"- clip seconds: `{clip_seconds}`",
        f"- cases analyzed: `{len(case_rows)}`",
        f"- mean law packets per case: `{summary['mean_num_law_packets']:.3f}`",
        f"- mean law families per case: `{summary['mean_num_law_families']:.3f}`",
        f"- aggregate families: `{aggregate_library['num_families']}`",
        f"- effective law families: `{summary['aggregate_law_effective_family_count']:.3f}`",
        f"- law family entropy: `{summary['aggregate_law_family_entropy']:.4f}`",
        f"- dominant law family pressure: `{summary['aggregate_law_dominant_family_pressure']:.4f}`",
        "",
        "## Top Families",
        "",
    ]
    for family in aggregate_library.get("families", []):
        q_mass = family["prototype_q_mass"]
        top_q_idx = max(range(len(q_mass)), key=lambda idx: q_mass[idx]) if q_mass else 0
        top_q = circle_cfg.qset[top_q_idx] if q_mass else None
        md_lines.extend(
            [
                f"### Family {family['family_index']}",
                f"- count: `{family['count']}`",
                f"- top q: `{top_q}`",
                f"- mean score: `{family['mean_score']:.4f}`",
                f"- mean major mass: `{family['mean_major_mass']:.4f}`",
                f"- mean minor residue: `{family['mean_minor_residue']:.4f}`",
                f"- mean promotability: `{family['mean_promotability']:.4f}`",
                f"- best similarity: `{family['best_similarity']:.4f}`",
                "",
            ]
        )
    (out_dir / "LAW_TOKEN_LIBRARY.md").write_text("\n".join(md_lines), encoding="utf-8")

    rel_md_lines = [
        "# Circleworld Relational Signature Library",
        "",
        f"- config: `{config_path}`",
        f"- cases: `{cases_path}`",
        f"- device: `{device}`",
        f"- run mode: `{run_mode}`",
        f"- clip seconds: `{clip_seconds}`",
        f"- mean signatures per case: `{summary['mean_num_relational_signatures']:.3f}`",
        f"- mean signature families per case: `{summary['mean_num_relational_signature_families']:.3f}`",
        f"- mean signature confidence: `{summary['mean_relational_signature_confidence']:.4f}`",
        f"- aggregate signature families: `{aggregate_signature_library['num_families']}`",
        f"- effective signature families: `{summary['aggregate_relational_effective_family_count']:.3f}`",
        f"- signature family entropy: `{summary['aggregate_relational_family_entropy']:.4f}`",
        f"- dominant signature family pressure: `{summary['aggregate_relational_dominant_family_pressure']:.4f}`",
        f"- case signature-family std: `{summary['case_relational_signature_family_count_std']:.4f}`",
        f"- case signature-confidence std: `{summary['case_relational_signature_confidence_std']:.4f}`",
        f"- case branch-mass std: `{summary['case_relational_branch_mass_std']:.4f}`",
        "",
        "## Top Signature Families",
        "",
    ]
    for family in aggregate_signature_library.get("families", []):
        q_profile = family["prototype_q_profile"]
        top_q_idx = max(range(len(q_profile)), key=lambda idx: q_profile[idx]) if q_profile else 0
        top_q = circle_cfg.qset[top_q_idx] if q_profile else None
        rel_md_lines.extend(
            [
                f"### Family {family['family_index']}",
                f"- count: `{family['count']}`",
                f"- top q: `{top_q}`",
                f"- mean confidence: `{family['mean_confidence']:.4f}`",
                f"- mean branch mass: `{family['mean_branch_mass']:.4f}`",
                f"- mean q entropy: `{family['mean_q_entropy']:.4f}`",
                f"- best similarity: `{family['best_similarity']:.4f}`",
                "",
            ]
        )
    (out_dir / "RELATIONAL_SIGNATURE_LIBRARY.md").write_text("\n".join(rel_md_lines), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Circleworld law-token library from real anchor audio.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases-json", default=str(_default_cases_path()))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=int, default=10)
    args = ap.parse_args()

    summary = build_library(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        cases_path=Path(args.cases_json),
        device_name=args.device,
        clip_seconds=int(args.clip_seconds),
    )
    print(json.dumps(
        {
            "saved": str(Path(args.out_dir) / "law_token_library.json"),
            "num_cases": summary["num_cases"],
            "mean_num_law_packets": summary["mean_num_law_packets"],
            "mean_num_law_families": summary["mean_num_law_families"],
            "aggregate_num_families": summary["aggregate_library"]["num_families"],
            "mean_num_relational_signatures": summary["mean_num_relational_signatures"],
            "aggregate_num_signature_families": summary["aggregate_relational_signature_library"]["num_families"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
