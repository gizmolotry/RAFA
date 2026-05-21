from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import load_config
from evaluate_relational_metamers import (  # reuses Circleworld metamer/run harness primitives
    _apply_transform,
    _default_cases_path,
    _load_cases,
    _run_circleworld,
    _safe_device,
    _transform_specs,
)
from export_circleworld_audio import _load_circle_cfg, prepare_reference_audio
from rafa_math_tools import phase_to_phasor
from stft_utils import compute_stft

SCHEMA = "rafa_dense_signature_claim_v0"
DEFAULT_CLUSTER_THRESHOLD = 0.92
EPS = 1.0e-8


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _select_cases(cases: dict[str, str], num_cases: int | None) -> dict[str, str]:
    if num_cases is None or int(num_cases) <= 0:
        return dict(cases)
    return dict(list(cases.items())[: int(num_cases)])


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(value):
            return float(value.detach().mean().item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return float(default)


def _mean(values: list[float]) -> float:
    return float(sum(float(v) for v in values) / max(1, len(values)))


def _stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return float((sum((float(v) - avg) ** 2 for v in values) / float(len(values))) ** 0.5)


def _safe_normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    if rows.numel() == 0:
        return rows.float()
    return F.normalize(rows.float(), dim=-1)


def _pooled_rows(rows: torch.Tensor | None) -> torch.Tensor | None:
    if rows is None or rows.numel() == 0:
        return None
    flat = rows.float().reshape(rows.size(0), -1)
    pooled = flat.mean(dim=0, keepdim=True)
    return F.normalize(pooled, dim=-1).squeeze(0)


def _cos(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        return 0.0
    aa = a.float().reshape(-1)
    bb = b.float().reshape(-1)
    size = min(int(aa.numel()), int(bb.numel()))
    if size <= 0:
        return 0.0
    return float(F.cosine_similarity(aa[:size].view(1, -1), bb[:size].view(1, -1), dim=-1).item())


def _pairwise_summary(rows: torch.Tensor, near_duplicate_threshold: float) -> dict[str, float]:
    if rows.numel() == 0 or rows.size(0) < 2:
        return {
            "mean_pairwise_cosine": 0.0,
            "mean_pairwise_distance": 0.0,
            "min_pairwise_distance": 0.0,
            "near_duplicate_pair_share": 0.0,
        }
    x = _safe_normalize_rows(rows.reshape(rows.size(0), -1))
    sim = torch.mm(x, x.t()).clamp(-1.0, 1.0)
    tri = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
    vals = sim[tri[0], tri[1]]
    dist = 1.0 - vals
    return {
        "mean_pairwise_cosine": float(vals.mean().item()),
        "mean_pairwise_distance": float(dist.mean().item()),
        "min_pairwise_distance": float(dist.min().item()),
        "near_duplicate_pair_share": float((vals >= float(near_duplicate_threshold)).float().mean().item()),
    }


def _effective_rank(rows: torch.Tensor) -> float:
    if rows.numel() == 0 or rows.size(0) < 2:
        return 0.0
    x = _safe_normalize_rows(rows.reshape(rows.size(0), -1))
    x = x - x.mean(dim=0, keepdim=True)
    try:
        singular = torch.linalg.svdvals(x)
    except RuntimeError:
        return 0.0
    total = singular.sum().clamp_min(EPS)
    probs = singular / total
    entropy = -(probs * probs.clamp_min(EPS).log()).sum()
    return float(torch.exp(entropy).item())


def _cluster_summary(rows: torch.Tensor, threshold: float) -> dict[str, float]:
    if rows.numel() == 0:
        return {
            "num_clusters": 0.0,
            "dominant_cluster_share": 0.0,
            "effective_cluster_count": 0.0,
        }
    x = _safe_normalize_rows(rows.reshape(rows.size(0), -1))
    prototypes: list[torch.Tensor] = []
    counts: list[int] = []
    for row in x:
        best_idx = -1
        best_sim = -1.0
        for idx, proto in enumerate(prototypes):
            sim = float(F.cosine_similarity(row.view(1, -1), proto.view(1, -1), dim=-1).item())
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim >= float(threshold):
            counts[best_idx] += 1
            weight_old = float(counts[best_idx] - 1) / float(counts[best_idx])
            weight_new = 1.0 / float(counts[best_idx])
            prototypes[best_idx] = F.normalize((weight_old * prototypes[best_idx] + weight_new * row).view(1, -1), dim=-1).squeeze(0)
        else:
            prototypes.append(row.clone())
            counts.append(1)
    total = float(sum(counts))
    probs = [float(count) / max(EPS, total) for count in counts]
    entropy = -sum(p * math.log(max(EPS, p)) for p in probs)
    return {
        "num_clusters": float(len(counts)),
        "dominant_cluster_share": float(max(probs) if probs else 0.0),
        "effective_cluster_count": float(math.exp(entropy) if probs else 0.0),
    }


def _representation_summary(rows: torch.Tensor, threshold: float) -> dict[str, float]:
    flat = rows.float().reshape(rows.size(0), -1) if rows.numel() else rows.float().reshape(0, 0)
    pairwise = _pairwise_summary(flat, near_duplicate_threshold=threshold)
    cluster = _cluster_summary(flat, threshold=threshold)
    return {
        "count": float(flat.size(0)),
        "dim": float(flat.size(1)) if flat.dim() == 2 else 0.0,
        "effective_rank": _effective_rank(flat),
        **pairwise,
        **cluster,
    }


def _empty_rows(device: torch.device | None = None) -> torch.Tensor:
    return torch.zeros(0, 0, device=device or torch.device("cpu"))


def _dense_rows(run: dict[str, Any]) -> torch.Tensor:
    bank = run.get("relational_signatures")
    if bank is None or getattr(bank, "num_signatures", 0) <= 0:
        return _empty_rows()
    return bank.h.detach().float()


def _metadata_rows(run: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packets = list(run.get("law_packets", []))
    rows: list[torch.Tensor] = []
    q_rows: list[torch.Tensor] = []
    law_rows: list[torch.Tensor] = []
    for packet in packets:
        q = packet.get("q_mass")
        law = packet.get("law_signature")
        if q is None or law is None:
            continue
        qv = q.detach().float().view(-1)
        qv = qv / qv.sum().clamp_min(EPS)
        lawv = law.detach().float().view(-1)
        lawv = F.normalize(lawv.view(1, -1), dim=-1).squeeze(0)
        rows.append(F.normalize(torch.cat([qv, lawv], dim=0).view(1, -1), dim=-1).squeeze(0))
        q_rows.append(qv)
        law_rows.append(lawv)
    if not rows:
        return _empty_rows(), _empty_rows(), _empty_rows()
    return torch.stack(rows, dim=0), torch.stack(q_rows, dim=0), torch.stack(law_rows, dim=0)


def _dense_prefix_metrics(base_rows: torch.Tensor, transformed_rows: torch.Tensor, prefix_dims: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {"prefix_cosines": {}, "mean_prefix_cos": 0.0}
    if base_rows.numel() == 0 or transformed_rows.numel() == 0:
        return out
    base_pool = _pooled_rows(base_rows)
    transformed_pool = _pooled_rows(transformed_rows)
    values: list[float] = []
    for dim in prefix_dims:
        use_dim = min(int(dim), int(base_rows.size(-1)), int(transformed_rows.size(-1)))
        if use_dim <= 0 or base_pool is None or transformed_pool is None:
            value = 0.0
        else:
            value = _cos(base_pool[:use_dim], transformed_pool[:use_dim])
        out["prefix_cosines"][str(use_dim)] = value
        values.append(value)
    out["mean_prefix_cos"] = _mean(values)
    return out


def _run_case(
    *,
    case_name: str,
    wav_path: Path,
    cfg: Any,
    stft_cfg: dict[str, Any],
    device: torch.device,
    device_name: str,
    clip_seconds: int,
    transform_specs: list[dict[str, Any]],
    cluster_threshold: float,
) -> dict[str, Any]:
    wav, _sr = prepare_reference_audio(
        wav_path=wav_path,
        device_name=device_name,
        clip_seconds_override=clip_seconds,
    )
    wav = wav.to(device)
    _mag, phase = compute_stft(wav, stft_cfg)
    phase_state = phase_to_phasor(phase)
    base_run = _run_circleworld(phase_state, cfg)
    base_bank = base_run.get("relational_signatures")
    prefix_dims = tuple(int(dim) for dim in getattr(base_bank, "prefix_dims", (128, 256, 384, 512, 640, 768)))

    dense_base = _dense_rows(base_run).to(device)
    metadata_base, metadata_q_base, metadata_law_base = _metadata_rows(base_run)
    metadata_base = metadata_base.to(device)
    metadata_q_base = metadata_q_base.to(device)
    metadata_law_base = metadata_law_base.to(device)
    dense_base_pool = _pooled_rows(dense_base)
    metadata_base_pool = _pooled_rows(metadata_base)
    metadata_q_base_pool = _pooled_rows(metadata_q_base)
    metadata_law_base_pool = _pooled_rows(metadata_law_base)

    transform_rows: list[dict[str, Any]] = []
    for transform in transform_specs:
        phase_state_t = _apply_transform(wav=wav, phase_state=phase_state, transform=transform, stft_cfg=stft_cfg)
        run_t = _run_circleworld(phase_state_t, cfg)
        dense_t = _dense_rows(run_t).to(device)
        metadata_t, metadata_q_t, metadata_law_t = _metadata_rows(run_t)
        metadata_t = metadata_t.to(device)
        metadata_q_t = metadata_q_t.to(device)
        metadata_law_t = metadata_law_t.to(device)
        dense_prefix = _dense_prefix_metrics(dense_base, dense_t, prefix_dims)
        dense_pooled_cos = _cos(dense_base_pool, _pooled_rows(dense_t))
        metadata_pooled_cos = _cos(metadata_base_pool, _pooled_rows(metadata_t))
        metadata_q_cos = _cos(metadata_q_base_pool, _pooled_rows(metadata_q_t))
        metadata_law_cos = _cos(metadata_law_base_pool, _pooled_rows(metadata_law_t))
        dense_stability_score = _mean([dense_pooled_cos, float(dense_prefix["mean_prefix_cos"])])
        metadata_stability_score = _mean([metadata_pooled_cos, metadata_q_cos, metadata_law_cos])
        transform_rows.append(
            {
                "transform": str(transform["name"]),
                "transform_family": str(transform.get("family", transform.get("kind", "unknown"))),
                "spec": transform,
                "dense": {
                    "pooled_full_cos": dense_pooled_cos,
                    "mean_prefix_cos": float(dense_prefix["mean_prefix_cos"]),
                    "prefix_cosines": dense_prefix["prefix_cosines"],
                    "stability_score": dense_stability_score,
                    "packet_count_delta": float(dense_t.size(0) - dense_base.size(0)),
                },
                "metadata": {
                    "pooled_full_cos": metadata_pooled_cos,
                    "q_profile_cos": metadata_q_cos,
                    "law_signature_cos": metadata_law_cos,
                    "stability_score": metadata_stability_score,
                    "packet_count_delta": float(metadata_t.size(0) - metadata_base.size(0)),
                },
                "delta_dense_minus_metadata_stability": float(dense_stability_score - metadata_stability_score),
                "winner": "dense" if dense_stability_score > metadata_stability_score else "metadata" if metadata_stability_score > dense_stability_score else "tie",
            }
        )

    dense_stability_values = [float(row["dense"]["stability_score"]) for row in transform_rows]
    metadata_stability_values = [float(row["metadata"]["stability_score"]) for row in transform_rows]
    case_summary = {
        "case": case_name,
        "wav_path": str(wav_path),
        "base": {
            "num_dense_signatures": int(dense_base.size(0)),
            "num_metadata_packets": int(metadata_base.size(0)),
            "dense": _representation_summary(dense_base.detach().cpu(), threshold=cluster_threshold),
            "metadata": _representation_summary(metadata_base.detach().cpu(), threshold=cluster_threshold),
        },
        "stability": {
            "dense_mean_stability_score": _mean(dense_stability_values),
            "metadata_mean_stability_score": _mean(metadata_stability_values),
            "delta_dense_minus_metadata": _mean(dense_stability_values) - _mean(metadata_stability_values),
            "dense_std_stability_score": _stddev(dense_stability_values),
            "metadata_std_stability_score": _stddev(metadata_stability_values),
            "dense_transform_wins": sum(1 for row in transform_rows if row["winner"] == "dense"),
            "metadata_transform_wins": sum(1 for row in transform_rows if row["winner"] == "metadata"),
            "tie_transform_wins": sum(1 for row in transform_rows if row["winner"] == "tie"),
        },
        "transforms": transform_rows,
    }
    return {
        **case_summary,
        "_dense_base_pool": dense_base_pool.detach().cpu() if dense_base_pool is not None else None,
        "_metadata_base_pool": metadata_base_pool.detach().cpu() if metadata_base_pool is not None else None,
        "_dense_packet_rows": dense_base.detach().cpu(),
        "_metadata_packet_rows": metadata_base.detach().cpu(),
    }


def _strip_internal_case(case: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if not key.startswith("_")}


def _stack_optional(rows: list[torch.Tensor | None]) -> torch.Tensor:
    kept = [row.float().view(-1) for row in rows if row is not None and row.numel() > 0]
    if not kept:
        return _empty_rows()
    min_dim = min(int(row.numel()) for row in kept)
    return torch.stack([row[:min_dim] for row in kept], dim=0)


def _concat_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    kept = [row.float().reshape(row.size(0), -1) for row in rows if row.numel() > 0 and row.size(0) > 0]
    if not kept:
        return _empty_rows()
    min_dim = min(int(row.size(1)) for row in kept)
    return torch.cat([row[:, :min_dim] for row in kept], dim=0)


def _aggregate_cases(cases: list[dict[str, Any]], cluster_threshold: float) -> dict[str, Any]:
    dense_case_rows = _stack_optional([case.get("_dense_base_pool") for case in cases])
    metadata_case_rows = _stack_optional([case.get("_metadata_base_pool") for case in cases])
    dense_packet_rows = _concat_rows([case["_dense_packet_rows"] for case in cases])
    metadata_packet_rows = _concat_rows([case["_metadata_packet_rows"] for case in cases])

    dense_stability = [float(case["stability"]["dense_mean_stability_score"]) for case in cases]
    metadata_stability = [float(case["stability"]["metadata_mean_stability_score"]) for case in cases]
    dense_transform_wins = sum(int(case["stability"]["dense_transform_wins"]) for case in cases)
    metadata_transform_wins = sum(int(case["stability"]["metadata_transform_wins"]) for case in cases)
    tie_transform_wins = sum(int(case["stability"]["tie_transform_wins"]) for case in cases)

    dense_sep = _pairwise_summary(dense_case_rows, near_duplicate_threshold=cluster_threshold)
    metadata_sep = _pairwise_summary(metadata_case_rows, near_duplicate_threshold=cluster_threshold)
    dense_collapse = _representation_summary(dense_packet_rows, threshold=cluster_threshold)
    metadata_collapse = _representation_summary(metadata_packet_rows, threshold=cluster_threshold)

    dense_stability_score = _mean(dense_stability)
    metadata_stability_score = _mean(metadata_stability)
    dense_separation_score = float(dense_sep["mean_pairwise_distance"])
    metadata_separation_score = float(metadata_sep["mean_pairwise_distance"])
    return {
        "num_cases": len(cases),
        "stability": {
            "dense_score": dense_stability_score,
            "metadata_score": metadata_stability_score,
            "delta_dense_minus_metadata": dense_stability_score - metadata_stability_score,
            "dense_std": _stddev(dense_stability),
            "metadata_std": _stddev(metadata_stability),
            "dense_transform_wins": dense_transform_wins,
            "metadata_transform_wins": metadata_transform_wins,
            "tie_transform_wins": tie_transform_wins,
        },
        "separation": {
            "dense_score": dense_separation_score,
            "metadata_score": metadata_separation_score,
            "delta_dense_minus_metadata": dense_separation_score - metadata_separation_score,
            "dense": dense_sep,
            "metadata": metadata_sep,
        },
        "collapse": {
            "dense": dense_collapse,
            "metadata": metadata_collapse,
            "delta_dense_minus_metadata": {
                "dominant_cluster_share": float(dense_collapse["dominant_cluster_share"] - metadata_collapse["dominant_cluster_share"]),
                "near_duplicate_pair_share": float(dense_collapse["near_duplicate_pair_share"] - metadata_collapse["near_duplicate_pair_share"]),
                "effective_rank": float(dense_collapse["effective_rank"] - metadata_collapse["effective_rank"]),
                "mean_pairwise_distance": float(dense_collapse["mean_pairwise_distance"] - metadata_collapse["mean_pairwise_distance"]),
                "effective_cluster_count": float(dense_collapse["effective_cluster_count"] - metadata_collapse["effective_cluster_count"]),
            },
        },
    }


def _verdict(aggregate: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    num_cases = int(aggregate.get("num_cases", 0))
    stability_delta = float(aggregate["stability"]["delta_dense_minus_metadata"])
    separation_delta = float(aggregate["separation"]["delta_dense_minus_metadata"])
    collapse_delta = aggregate["collapse"]["delta_dense_minus_metadata"]
    dense_collapse = aggregate["collapse"]["dense"]
    metadata_collapse = aggregate["collapse"]["metadata"]

    enough_cases = num_cases >= 2
    stable_better = stability_delta > 1.0e-4
    separated_better = separation_delta > 1.0e-4
    no_worse_dominant = float(collapse_delta["dominant_cluster_share"]) <= 0.02
    no_worse_duplicates = float(collapse_delta["near_duplicate_pair_share"]) <= 0.02
    no_worse_rank = float(collapse_delta["effective_rank"]) >= -0.05
    no_worse_packet_diversity = float(collapse_delta["mean_pairwise_distance"]) >= -0.02
    no_worse_collapse = no_worse_dominant and no_worse_duplicates and no_worse_rank and no_worse_packet_diversity

    if not enough_cases:
        reasons.append("At least two cases are required for an aggregate separation claim.")
    if not stable_better:
        reasons.append("Dense does not beat explicit metadata on aggregate stability score.")
    if not separated_better:
        reasons.append("Dense does not beat explicit metadata on aggregate case separation.")
    if not no_worse_collapse:
        reasons.append("Dense has worse collapse diagnostics on at least one aggregate anti-collapse metric.")
    no_data = float(dense_collapse.get("count", 0.0)) <= 0.0 or float(metadata_collapse.get("count", 0.0)) <= 0.0
    if no_data:
        reasons.append("One or both representations produced no packet rows.")

    dense_wins = enough_cases and stable_better and separated_better and no_worse_collapse and not no_data
    if dense_wins:
        status = "dense_wins"
        reasons.append("Dense beats metadata on aggregate stability and separation without worse collapse diagnostics.")
    elif enough_cases and not no_data and no_worse_collapse:
        status = "open"
    else:
        status = "needs_review"
    return {
        "status": status,
        "dense_wins": bool(dense_wins),
        "criteria": {
            "enough_cases": enough_cases,
            "stable_better": stable_better,
            "separated_better": separated_better,
            "no_worse_collapse": no_worse_collapse,
            "no_worse_dominant_cluster_share": no_worse_dominant,
            "no_worse_near_duplicate_pair_share": no_worse_duplicates,
            "no_worse_effective_rank": no_worse_rank,
            "no_worse_packet_pairwise_distance": no_worse_packet_diversity,
        },
        "reasons": reasons,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    verdict = summary["verdict"]
    md_lines = [
        "# Dense Signature Claim Harness",
        "",
        f"- schema: `{summary['schema']}`",
        f"- config: `{summary['config_path']}`",
        f"- cases: `{summary['cases_path']}`",
        f"- device: `{summary['device']}`",
        f"- clip seconds: `{summary['clip_seconds']}`",
        f"- verdict: `{verdict['status']}`",
        "",
        "## Aggregate",
        "",
        f"- dense stability score: `{aggregate['stability']['dense_score']:.6f}`",
        f"- metadata stability score: `{aggregate['stability']['metadata_score']:.6f}`",
        f"- stability delta dense-minus-metadata: `{aggregate['stability']['delta_dense_minus_metadata']:+.6f}`",
        f"- dense separation score: `{aggregate['separation']['dense_score']:.6f}`",
        f"- metadata separation score: `{aggregate['separation']['metadata_score']:.6f}`",
        f"- separation delta dense-minus-metadata: `{aggregate['separation']['delta_dense_minus_metadata']:+.6f}`",
        f"- dense dominant cluster share: `{aggregate['collapse']['dense']['dominant_cluster_share']:.6f}`",
        f"- metadata dominant cluster share: `{aggregate['collapse']['metadata']['dominant_cluster_share']:.6f}`",
        f"- dense effective rank: `{aggregate['collapse']['dense']['effective_rank']:.6f}`",
        f"- metadata effective rank: `{aggregate['collapse']['metadata']['effective_rank']:.6f}`",
        f"- dense near-duplicate pair share: `{aggregate['collapse']['dense']['near_duplicate_pair_share']:.6f}`",
        f"- metadata near-duplicate pair share: `{aggregate['collapse']['metadata']['near_duplicate_pair_share']:.6f}`",
        "",
        "## Verdict Reasons",
        "",
    ]
    md_lines.extend([f"- {reason}" for reason in verdict.get("reasons", [])] or ["- No verdict reason recorded."])
    md_lines.extend(["", "## Per Case", ""])
    for case in summary.get("cases", []):
        stability = case["stability"]
        base = case["base"]
        md_lines.extend(
            [
                f"### {case['case']}",
                f"- dense signatures: `{base['num_dense_signatures']}`",
                f"- metadata packets: `{base['num_metadata_packets']}`",
                f"- dense stability: `{stability['dense_mean_stability_score']:.6f}`",
                f"- metadata stability: `{stability['metadata_mean_stability_score']:.6f}`",
                f"- delta dense-minus-metadata: `{stability['delta_dense_minus_metadata']:+.6f}`",
                f"- transform wins dense/metadata/tie: `{stability['dense_transform_wins']}` / `{stability['metadata_transform_wins']}` / `{stability['tie_transform_wins']}`",
                "",
            ]
        )
    path.write_text("\n".join(md_lines), encoding="utf-8")


def evaluate_dense_signature_claim(
    *,
    config_path: Path,
    out_dir: Path,
    cases_path: Path,
    device_name: str,
    num_cases: int | None,
    clip_seconds: int,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    cfg = _load_circle_cfg(config_path)
    stft_cfg = load_config()["data"]["stft"]
    all_cases = _load_cases(cases_path)
    selected_cases = _select_cases(all_cases, num_cases)
    transforms = _transform_specs()
    threshold = float(getattr(cfg, "law_packet_merge_threshold", DEFAULT_CLUSTER_THRESHOLD))

    out_dir.mkdir(parents=True, exist_ok=True)
    case_rows_internal: list[dict[str, Any]] = []
    for case_name, wav_path_str in selected_cases.items():
        case_rows_internal.append(
            _run_case(
                case_name=case_name,
                wav_path=Path(wav_path_str),
                cfg=cfg,
                stft_cfg=stft_cfg,
                device=device,
                device_name=device_name,
                clip_seconds=clip_seconds,
                transform_specs=transforms,
                cluster_threshold=threshold,
            )
        )

    aggregate = _aggregate_cases(case_rows_internal, cluster_threshold=threshold)
    summary = {
        "schema": SCHEMA,
        "runtime": "circleworld_proto",
        "created_utc": _utc_timestamp(),
        "config_path": str(config_path),
        "cases_path": str(cases_path),
        "device": str(device),
        "requested_device": str(device_name),
        "num_cases_requested": int(num_cases) if num_cases is not None else None,
        "num_cases_available": len(all_cases),
        "num_cases": len(case_rows_internal),
        "clip_seconds": int(clip_seconds),
        "cluster_threshold": threshold,
        "representation_definitions": {
            "dense": "pooled RAFA relational signature bank h, with prefix cosine metrics from h prefixes",
            "metadata": "explicit law-token metadata vector built from each packet q_mass and law_signature fields",
        },
        "transform_specs": transforms,
        "aggregate": aggregate,
        "cases": [_strip_internal_case(case) for case in case_rows_internal],
    }
    summary["verdict"] = _verdict(aggregate)
    json_path = out_dir / "dense_signature_claim.json"
    md_path = out_dir / "DENSE_SIGNATURE_CLAIM.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate dense RAFA signatures against explicit q/law metadata on the same Circleworld packet streams.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=0)
    ap.add_argument("--clip-seconds", type=int, default=4)
    ap.add_argument("--cases-json", default=str(_default_cases_path()))
    args = ap.parse_args()

    summary = evaluate_dense_signature_claim(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        cases_path=Path(args.cases_json),
        device_name=str(args.device),
        num_cases=int(args.num_cases),
        clip_seconds=int(args.clip_seconds),
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "saved": str(Path(args.out_dir) / "dense_signature_claim.json"),
                "report": str(Path(args.out_dir) / "DENSE_SIGNATURE_CLAIM.md"),
                "num_cases": summary["num_cases"],
                "verdict": summary["verdict"],
                "aggregate": {
                    "stability_delta_dense_minus_metadata": summary["aggregate"]["stability"]["delta_dense_minus_metadata"],
                    "separation_delta_dense_minus_metadata": summary["aggregate"]["separation"]["delta_dense_minus_metadata"],
                    "collapse_delta_dense_minus_metadata": summary["aggregate"]["collapse"]["delta_dense_minus_metadata"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
