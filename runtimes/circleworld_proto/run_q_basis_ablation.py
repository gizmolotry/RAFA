from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_circleworld import evaluate_config


DEFAULT_BASE_CONFIG = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "relsig_scout_balance_2026-05-04"
    / "circleworld_real_anchor_config_cem_v1.json"
)

VARIANT_CONFIG_NAME = "config.json"
SUMMARY_JSON_NAME = "q_basis_ablation_summary.json"
SUMMARY_MD_NAME = "Q_BASIS_ABLATION_SUMMARY.md"

PARENT_BRANCH_METRICS = (
    "mean_real_branch_fraction",
    "mean_parent_real_branch_fraction",
)
CHILD_BRANCH_METRICS = (
    "mean_child_real_branch_fraction",
    "mean_child_world_count",
    "mean_live_child_fraction",
    "mean_child_writeback_mass",
    "mean_child_writeback_gate_mass",
    "mean_child_phase_writeback_delta_mass",
    "mean_child_parent_phase_writeback_delta_mass",
    "mean_child_support_writeback_mass",
    "mean_child_logit_writeback_mass",
    "mean_child_qtrace_writeback_mass",
    "mean_child_parent_divergence",
    "mean_child_sibling_divergence",
)
PHASE_ONLY_METRICS = (
    "phase_only_real_branch_fraction",
    "phase_only_excess_branch_fraction",
    "decorative_slot2_low_phase_fraction",
    "mean_phase_only_branch_surface",
    "mean_phase_only_branch_phase_wall",
    "mean_phase_only_branch_distinctness",
    "mean_phase_only_branch_distinctness_balanced",
)
LAW_SIGNATURE_METRICS = (
    "mean_num_law_packets",
    "mean_num_law_families",
    "mean_law_family_entropy",
    "mean_dominant_law_family_share",
    "mean_dominant_law_q_share",
    "mean_num_law_top_q_unique",
    "mean_num_relational_signatures",
    "mean_num_relational_signature_families",
    "mean_relational_signature_confidence",
    "mean_relational_signature_q_entropy",
    "mean_relational_branch_mass",
    "mean_dominant_relational_family_share",
)
BENCHMARKISH_HELDOUT_METRICS = (
    "mean_major_gain",
    "mean_residue_drop",
    "mean_promotability_gain",
    "mean_dominant_q_share",
    "mean_q_entropy",
    "mean_major_saturation",
    "mean_loss",
    "mean_l_q_dom",
    "mean_l_q_entropy",
    "mean_l_major_sat",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _config_block(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config")
    if isinstance(cfg, dict):
        return cfg
    return payload


def _variant_payload(base_payload: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(base_payload)
    _config_block(payload).update(overrides)
    return payload


def _mean(values: list[float], fallback: float = 1.0) -> float:
    if not values:
        return fallback
    return float(sum(values) / len(values))


def _shuffled_qset(qset: list[int], seed: int) -> list[int]:
    shuffled = list(qset)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if shuffled == qset and len(shuffled) > 1:
        shuffled = shuffled[1:] + shuffled[:1]
    return shuffled


def _compact_fourierish(qset: list[int], q_weights: list[float]) -> tuple[list[int], list[float], str]:
    wanted = [2, 4, 8, 12]
    available = {int(q): idx for idx, q in enumerate(qset)}
    compact = [q for q in wanted if q in available]
    note = "power-of-two-plus-octave subset from existing qset"
    if len(compact) < 3:
        compact = sorted(qset)[: min(4, len(qset))]
        note = "fallback compact low-q subset; preferred [2,4,8,12] was not available"
    weights = [float(q_weights[available[q]]) for q in compact]
    avg = _mean(weights, fallback=_mean(q_weights))
    weights = [avg for _ in compact]
    return compact, weights, note


def _variant_specs(base_cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    qset = [int(q) for q in base_cfg.get("qset", (2, 3, 4, 5, 6, 8, 12))]
    q_weights = [float(w) for w in base_cfg.get("q_weights", (1.0,) * len(qset))]
    uniform_weight = _mean(q_weights)
    compact_qset, compact_weights, compact_note = _compact_fourierish(qset, q_weights)
    shuffled = _shuffled_qset(qset, seed)

    return [
        {
            "name": "ramanujan_baseline",
            "description": "Ramanujan branch kernel with the base q-basis and learned weights.",
            "overrides": {
                "branch_kernel_version": "ramanujan",
                "qset": qset,
                "q_weights": q_weights,
            },
        },
        {
            "name": "qtrace_only_control",
            "description": "Removes Ramanujan/per-q phase alignment from the relation kernel.",
            "overrides": {
                "branch_kernel_version": "qtrace_only",
                "qset": qset,
                "q_weights": q_weights,
            },
        },
        {
            "name": "uniform_q_weights",
            "description": "Keeps qset fixed but flattens q weights to the base mean weight.",
            "overrides": {
                "branch_kernel_version": "ramanujan",
                "qset": qset,
                "q_weights": [uniform_weight for _ in q_weights],
            },
        },
        {
            "name": "shuffled_qset",
            "description": "Deterministically shuffles q labels while preserving the learned weight order.",
            "overrides": {
                "branch_kernel_version": "ramanujan",
                "qset": shuffled,
                "q_weights": q_weights,
            },
        },
        {
            "name": "compact_fourierish_control",
            "description": compact_note,
            "overrides": {
                "branch_kernel_version": "ramanujan",
                "qset": compact_qset,
                "q_weights": compact_weights,
                "q_trace_rank": min(int(base_cfg.get("q_trace_rank", 4)), len(compact_qset)),
            },
        },
        {
            "name": "compact_qtrace_only_control",
            "description": "Uses the compact Fourier-ish q subset but removes Ramanujan/per-q phase alignment.",
            "overrides": {
                "branch_kernel_version": "qtrace_only",
                "qset": compact_qset,
                "q_weights": compact_weights,
                "q_trace_rank": min(int(base_cfg.get("q_trace_rank", 4)), len(compact_qset)),
            },
        },
        {
            "name": "phase_only_relation_control",
            "description": "Disables q and qtrace relation kernels; branch relation uses phase alignment only.",
            "overrides": {
                "branch_kernel_version": "phase_only",
                "qset": qset,
                "q_weights": q_weights,
            },
        },
        {
            "name": "uniform_relation_kernel_control",
            "description": "Disables q, qtrace, and phase relation selectivity with a uniform relation kernel.",
            "overrides": {
                "branch_kernel_version": "uniform",
                "qset": qset,
                "q_weights": q_weights,
            },
        },
    ]


def _pick(source: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    block = source if isinstance(source, dict) else {}
    return {key: block.get(key) for key in keys}


def _source_split(heldout: dict[str, Any] | None) -> dict[str, Any]:
    by_source = (heldout or {}).get("by_source")
    by_source = by_source if isinstance(by_source, dict) else {}
    split: dict[str, Any] = {}
    keys = (
        ("count",)
        + PARENT_BRANCH_METRICS
        + CHILD_BRANCH_METRICS
        + PHASE_ONLY_METRICS
        + LAW_SIGNATURE_METRICS
        + BENCHMARKISH_HELDOUT_METRICS
    )
    for source in ("synthetic", "naked_rafa"):
        split[source] = _pick(by_source.get(source), keys)
    return split


def _extract_metrics(heldout: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "parent_branch": _pick(heldout, PARENT_BRANCH_METRICS),
        "child_branch": _pick(heldout, CHILD_BRANCH_METRICS),
        "phase_only_branch": _pick(heldout, PHASE_ONLY_METRICS),
        "law_signature_family": _pick(heldout, LAW_SIGNATURE_METRICS),
        "benchmarkish_heldout": _pick(heldout, BENCHMARKISH_HELDOUT_METRICS),
        "source_split": _source_split(heldout),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _flat_metric(row: dict[str, Any], section: str, key: str) -> Any:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    block = metrics.get(section)
    if not isinstance(block, dict):
        return None
    return block.get(key)


def _source_metric(row: dict[str, Any], source: str, key: str) -> Any:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    split = metrics.get("source_split")
    if not isinstance(split, dict):
        return None
    block = split.get(source)
    if not isinstance(block, dict):
        return None
    return block.get(key)


def _write_markdown(out_dir: Path, summary: dict[str, Any]) -> None:
    rows = list(summary.get("variants", []))
    lines = [
        "# Circleworld Q-Basis / Ramanujan Ablation",
        "",
        f"- base config: `{summary.get('base_config_path')}`",
        f"- device: `{summary.get('device')}`",
        f"- time steps: `{summary.get('time_steps')}`",
        f"- write only: `{summary.get('write_only')}`",
        "",
        "## Variant Configs",
        "",
        "| variant | kernel | qset | q_weights | config | status |",
        "|---|---:|---|---|---|---|",
    ]
    for row in rows:
        cfg = row.get("config", {})
        q_weights = cfg.get("q_weights") if isinstance(cfg, dict) else None
        q_weight_str = (
            "[" + ", ".join(_fmt(float(w)) for w in q_weights) + "]"
            if isinstance(q_weights, list)
            else _fmt(q_weights)
        )
        lines.append(
            "| {variant} | {kernel} | {qset} | {q_weights} | `{config_path}` | {status} |".format(
                variant=row.get("variant"),
                kernel=_fmt(cfg.get("branch_kernel_version") if isinstance(cfg, dict) else None),
                qset=_fmt(cfg.get("qset") if isinstance(cfg, dict) else None),
                q_weights=q_weight_str,
                config_path=row.get("config_path"),
                status=row.get("status"),
            )
        )

    lines.extend(
        [
            "",
            "## Heldout Comparison",
            "",
            "| variant | parent branch | child branch | phase-only branch | law families | relsig families | relsig conf | heldout loss | q entropy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {parent} | {child} | {phase} | {law_fam} | {sig_fam} | {sig_conf} | {loss} | {entropy} |".format(
                variant=row.get("variant"),
                parent=_fmt(_flat_metric(row, "parent_branch", "mean_parent_real_branch_fraction")),
                child=_fmt(_flat_metric(row, "child_branch", "mean_child_real_branch_fraction")),
                phase=_fmt(_flat_metric(row, "phase_only_branch", "phase_only_real_branch_fraction")),
                law_fam=_fmt(_flat_metric(row, "law_signature_family", "mean_num_law_families")),
                sig_fam=_fmt(_flat_metric(row, "law_signature_family", "mean_num_relational_signature_families")),
                sig_conf=_fmt(_flat_metric(row, "law_signature_family", "mean_relational_signature_confidence")),
                loss=_fmt(_flat_metric(row, "benchmarkish_heldout", "mean_loss")),
                entropy=_fmt(_flat_metric(row, "benchmarkish_heldout", "mean_q_entropy")),
            )
        )

    lines.extend(
        [
            "",
            "## Child Writeback Mechanism",
            "",
            "| variant | gate mass | phase delta | parent phase delta | support mass | logit mass | qtrace mass |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {gate} | {phase} | {parent_phase} | {support} | {logit} | {qtrace} |".format(
                variant=row.get("variant"),
                gate=_fmt(_flat_metric(row, "child_branch", "mean_child_writeback_gate_mass")),
                phase=_fmt(_flat_metric(row, "child_branch", "mean_child_phase_writeback_delta_mass")),
                parent_phase=_fmt(
                    _flat_metric(row, "child_branch", "mean_child_parent_phase_writeback_delta_mass")
                ),
                support=_fmt(_flat_metric(row, "child_branch", "mean_child_support_writeback_mass")),
                logit=_fmt(_flat_metric(row, "child_branch", "mean_child_logit_writeback_mass")),
                qtrace=_fmt(_flat_metric(row, "child_branch", "mean_child_qtrace_writeback_mass")),
            )
        )

    lines.extend(
        [
            "",
            "## Naked / Synthetic Split",
            "",
            "| variant | synthetic parent | synthetic child | synthetic phase-only | naked parent | naked child | naked phase-only |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {s_parent} | {s_child} | {s_phase} | {n_parent} | {n_child} | {n_phase} |".format(
                variant=row.get("variant"),
                s_parent=_fmt(_source_metric(row, "synthetic", "mean_parent_real_branch_fraction")),
                s_child=_fmt(_source_metric(row, "synthetic", "mean_child_real_branch_fraction")),
                s_phase=_fmt(_source_metric(row, "synthetic", "phase_only_real_branch_fraction")),
                n_parent=_fmt(_source_metric(row, "naked_rafa", "mean_parent_real_branch_fraction")),
                n_child=_fmt(_source_metric(row, "naked_rafa", "mean_child_real_branch_fraction")),
                n_phase=_fmt(_source_metric(row, "naked_rafa", "phase_only_real_branch_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `benchmarkish_heldout` metrics come from `evaluate_config`; this harness does not run audio rendering or training.",
            "- `naked_rafa` source rows are only produced by the existing evaluator when its heldout plan includes them.",
        ]
    )
    (out_dir / SUMMARY_MD_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_q_basis_ablation(
    *,
    out_dir: Path,
    base_config_path: Path,
    device_name: str,
    time_steps: int,
    write_only: bool,
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_payload = _load_json(base_config_path)
    base_cfg = _config_block(base_payload)

    rows: list[dict[str, Any]] = []
    for spec in _variant_specs(base_cfg, seed):
        variant_dir = out_dir / spec["name"]
        config_path = variant_dir / VARIANT_CONFIG_NAME
        payload = _variant_payload(base_payload, spec["overrides"])
        cfg = _config_block(payload)
        _write_json(config_path, payload)

        heldout = None
        status = "write_only"
        eval_dir = variant_dir / "heldout_eval"
        if not write_only:
            try:
                heldout = evaluate_config(
                    config_path=config_path,
                    out_dir=eval_dir,
                    time_steps=time_steps,
                    device_name=device_name,
                )
                status = "evaluated"
            except Exception as exc:
                status = "evaluation_failed"
                heldout = {"error": f"{type(exc).__name__}: {exc}"}

        rows.append(
            {
                "variant": spec["name"],
                "description": spec["description"],
                "status": status,
                "config_path": str(config_path),
                "heldout_summary": str(eval_dir / "heldout_summary.json") if status == "evaluated" else None,
                "config": {
                    "branch_kernel_version": cfg.get("branch_kernel_version"),
                    "qset": cfg.get("qset"),
                    "q_weights": cfg.get("q_weights"),
                    "q_trace_rank": cfg.get("q_trace_rank"),
                    "branching_mode": cfg.get("branching_mode"),
                    "branch_law_version": cfg.get("branch_law_version"),
                },
                "overrides": spec["overrides"],
                "metrics": _extract_metrics(heldout if status == "evaluated" else None),
                "error": heldout.get("error") if isinstance(heldout, dict) and "error" in heldout else None,
            }
        )

    summary = {
        "runtime": "circleworld_proto",
        "mode": "q_basis_ramanujan_ablation",
        "base_config_path": str(base_config_path),
        "out_dir": str(out_dir),
        "device": device_name,
        "time_steps": time_steps,
        "write_only": write_only,
        "seed": seed,
        "variants": rows,
    }
    _write_json(out_dir / SUMMARY_JSON_NAME, summary)
    _write_markdown(out_dir, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate and optionally evaluate Circleworld q-basis/Ramanujan ablation configs."
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-steps", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260506)
    ap.add_argument("--write-only", action="store_true", help="Write variant configs and reports without evaluation.")
    args = ap.parse_args()

    summary = run_q_basis_ablation(
        out_dir=Path(args.out_dir),
        base_config_path=Path(args.base_config),
        device_name=args.device,
        time_steps=args.time_steps,
        write_only=bool(args.write_only),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
