from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, RUNTIME, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import circleworld  # type: ignore  # Circleworld-local monkeypatch target.
from evaluate_circleworld import _load_circle_cfg, evaluate_circle_cfg, evaluate_config


DEFAULT_BASE_CONFIG = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "relsig_scout_balance_2026-05-04"
    / "circleworld_real_anchor_config_cem_v1.json"
)

VARIANT_CONFIG_NAME = "config.json"
SUMMARY_JSON_NAME = "substrate_ablation_summary.json"
SUMMARY_MD_NAME = "SUBSTRATE_ABLATION_SUMMARY.md"

BRANCH_METRICS = (
    "mean_real_branch_fraction",
    "mean_parent_real_branch_fraction",
    "mean_child_real_branch_fraction",
    "phase_only_real_branch_fraction",
)

DECORATIVE_EXCESS_METRICS = (
    "mean_decorative_slot2_fraction",
    "decorative_slot2_low_phase_fraction",
    "phase_only_excess_branch_fraction",
)

SUBSTRATE_MASK_METRICS = (
    "mean_topology_mask",
    "mean_complexity_mask",
    "mean_context_mask",
    "mean_branch_defect",
    "mean_branch_world_grad",
    "mean_branch_q_disagreement",
    "mean_branch_phase_wall",
)

BRANCH_PRESSURE_METRICS = (
    "mean_branch_positive_mask",
    "mean_branch_negative_mask",
)

BENCHMARK_METRICS = (
    "mean_major_gain",
    "mean_residue_drop",
    "mean_loss",
    "mean_l_q_dom",
    "mean_l_q_entropy",
    "mean_l_major_sat",
)

CHILD_METRICS = (
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

ALL_METRICS = (
    BRANCH_METRICS
    + DECORATIVE_EXCESS_METRICS
    + SUBSTRATE_MASK_METRICS
    + BRANCH_PRESSURE_METRICS
    + BENCHMARK_METRICS
    + CHILD_METRICS
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _config_block(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config")
    return cfg if isinstance(cfg, dict) else payload


def _variant_payload(base_payload: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(base_payload)
    _config_block(payload).update(overrides)
    return payload


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "lattice_baseline",
            "description": "Unmodified Circleworld lattice/neighborhood substrate.",
            "overrides": {},
            "smooth_patch": None,
        },
        {
            "name": "pointwise_no_neighborhood",
            "description": "Pointwise control with neighborhood smoothing and support spreading disabled by config.",
            "overrides": {"mask_neighborhood": 1, "support_spread": 1},
            "smooth_patch": None,
        },
        {
            "name": "shuffled_neighborhood",
            "description": "Deterministic flatten-permutation before smoothing, followed by unpermutation.",
            "overrides": {},
            "smooth_patch": "shuffled",
        },
        {
            "name": "flat_global_mean",
            "description": "Broadcasts each smoothed field to its global spatial mean.",
            "overrides": {},
            "smooth_patch": "global_mean",
        },
    ]


def _coprime_stride(size: int, seed: int) -> int:
    stride = max(1, int(seed) % max(1, size))
    while math.gcd(stride, size) != 1:
        stride += 1
        if stride >= size:
            stride = 1
    return stride


def _flatten_permutation(size: int, seed: int, device: Any) -> tuple[Any, Any]:
    import torch

    stride = _coprime_stride(size, seed)
    offset = (int(seed) * 104729 + 17) % size
    idx = torch.arange(size, device=device, dtype=torch.long)
    perm = (idx * stride + offset) % size
    inv = torch.empty_like(perm)
    inv[perm] = idx
    return perm, inv


@contextmanager
def _patched_smoothing(kind: str | None, seed: int) -> Iterator[dict[str, Any]]:
    original = getattr(circleworld, "_smooth_field_2d", None)
    meta: dict[str, Any] = {
        "requested": kind,
        "applied": False,
        "limitation": None,
    }
    if kind is None:
        yield meta
        return
    if original is None or not callable(original):
        meta["limitation"] = "circleworld._smooth_field_2d was not found or was not callable."
        yield meta
        return

    if kind == "shuffled":

        def shuffled_smooth(field: Any, kernel: int) -> Any:
            if getattr(field, "dim", lambda: 0)() != 3:
                return original(field, kernel)
            flat_size = int(field.size(1) * field.size(2))
            if flat_size <= 1:
                return original(field, kernel)
            perm, inv = _flatten_permutation(flat_size, seed, field.device)
            shuffled = field.reshape(field.size(0), flat_size).index_select(-1, perm).reshape_as(field)
            smoothed = original(shuffled, kernel)
            return smoothed.reshape(field.size(0), flat_size).index_select(-1, inv).reshape_as(field)

        replacement = shuffled_smooth
    elif kind == "global_mean":

        def global_mean_smooth(field: Any, kernel: int) -> Any:
            if getattr(field, "dim", lambda: 0)() >= 2:
                return field.mean(dim=tuple(range(1, field.dim())), keepdim=True).expand_as(field).clamp(0.0, 1.0)
            return field.mean().expand_as(field).clamp(0.0, 1.0)

        replacement = global_mean_smooth
    else:
        meta["limitation"] = f"Unknown smoothing patch kind: {kind}"
        yield meta
        return

    setattr(circleworld, "_smooth_field_2d", replacement)
    meta["applied"] = True
    try:
        yield meta
    finally:
        setattr(circleworld, "_smooth_field_2d", original)


def _synthetic_plan(seed_start: int, count: int) -> list[tuple[str, int]]:
    return [("synthetic", int(seed_start) + i) for i in range(max(1, int(count)))]


def _can_inject_plan() -> bool:
    try:
        sig = inspect.signature(evaluate_circle_cfg)
    except (TypeError, ValueError):
        return False
    return "plan" in sig.parameters


def _evaluate_variant(
    *,
    config_path: Path,
    eval_dir: Path,
    time_steps: int,
    device_name: str,
    plan: list[tuple[str, int]] | None,
) -> tuple[dict[str, Any], str | None]:
    if plan is not None and _can_inject_plan():
        cfg = _load_circle_cfg(config_path)
        return (
            evaluate_circle_cfg(
                cfg,
                time_steps=time_steps,
                device_name=device_name,
                plan=plan,
                config_path=config_path,
                out_dir=eval_dir,
                include_trajectory=True,
            ),
            None,
        )
    limitation = None
    if plan is not None:
        limitation = "evaluate_circle_cfg plan injection unavailable; used evaluate_config default heldout plan."
    return (
        evaluate_config(
            config_path=config_path,
            out_dir=eval_dir,
            time_steps=time_steps,
            device_name=device_name,
        ),
        limitation,
    )


def _pick(source: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    block = source if isinstance(source, dict) else {}
    return {key: block.get(key) for key in keys}


def _source_split(heldout: dict[str, Any] | None) -> dict[str, Any]:
    by_source = (heldout or {}).get("by_source")
    by_source = by_source if isinstance(by_source, dict) else {}
    return {
        source: _pick(by_source.get(source), ("count",) + ALL_METRICS)
        for source in ("synthetic", "naked_rafa")
    }


def _extract_metrics(heldout: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "branch": _pick(heldout, BRANCH_METRICS),
        "decorative_excess": _pick(heldout, DECORATIVE_EXCESS_METRICS),
        "substrate_masks": _pick(heldout, SUBSTRATE_MASK_METRICS),
        "branch_pressure": _pick(heldout, BRANCH_PRESSURE_METRICS),
        "major_residue_loss": _pick(heldout, BENCHMARK_METRICS),
        "child": _pick(heldout, CHILD_METRICS),
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


def _metric(row: dict[str, Any], section: str, key: str) -> Any:
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
        "# Circleworld Substrate / Lattice Ablation",
        "",
        f"- base config: `{summary.get('base_config_path')}`",
        f"- device: `{summary.get('device')}`",
        f"- time steps: `{summary.get('time_steps')}`",
        f"- write only: `{summary.get('write_only')}`",
        f"- synthetic only requested: `{summary.get('synthetic_only')}`",
        "",
        "## Variants",
        "",
        "| variant | overrides | smoothing patch | status | config |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | `{overrides}` | {patch} | {status} | `{config}` |".format(
                variant=row.get("variant"),
                overrides=json.dumps(row.get("overrides", {}), sort_keys=True),
                patch=_fmt(row.get("smooth_patch")),
                status=row.get("status"),
                config=row.get("config_path"),
            )
        )

    lines.extend(
        [
            "",
            "## Branch Metrics",
            "",
            "| variant | parent | child | phase-only | real total | decorative | excess |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {parent} | {child} | {phase} | {real} | {decorative} | {excess} |".format(
                variant=row.get("variant"),
                parent=_fmt(_metric(row, "branch", "mean_parent_real_branch_fraction")),
                child=_fmt(_metric(row, "branch", "mean_child_real_branch_fraction")),
                phase=_fmt(_metric(row, "branch", "phase_only_real_branch_fraction")),
                real=_fmt(_metric(row, "branch", "mean_real_branch_fraction")),
                decorative=_fmt(_metric(row, "decorative_excess", "mean_decorative_slot2_fraction")),
                excess=_fmt(_metric(row, "decorative_excess", "phase_only_excess_branch_fraction")),
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
            "| {variant} | {gate} | {phase_delta} | {parent_phase_delta} | {support} | {logit} | {qtrace} |".format(
                variant=row.get("variant"),
                gate=_fmt(_metric(row, "child", "mean_child_writeback_gate_mass")),
                phase_delta=_fmt(_metric(row, "child", "mean_child_phase_writeback_delta_mass")),
                parent_phase_delta=_fmt(_metric(row, "child", "mean_child_parent_phase_writeback_delta_mass")),
                support=_fmt(_metric(row, "child", "mean_child_support_writeback_mass")),
                logit=_fmt(_metric(row, "child", "mean_child_logit_writeback_mass")),
                qtrace=_fmt(_metric(row, "child", "mean_child_qtrace_writeback_mass")),
            )
        )

    lines.extend(
        [
            "",
            "## Substrate Masks",
            "",
            "| variant | topology | complexity | context | defect | q disagree | phase wall | pos | neg |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {topology} | {complexity} | {context} | {defect} | {q} | {wall} | {pos} | {neg} |".format(
                variant=row.get("variant"),
                topology=_fmt(_metric(row, "substrate_masks", "mean_topology_mask")),
                complexity=_fmt(_metric(row, "substrate_masks", "mean_complexity_mask")),
                context=_fmt(_metric(row, "substrate_masks", "mean_context_mask")),
                defect=_fmt(_metric(row, "substrate_masks", "mean_branch_defect")),
                q=_fmt(_metric(row, "substrate_masks", "mean_branch_q_disagreement")),
                wall=_fmt(_metric(row, "substrate_masks", "mean_branch_phase_wall")),
                pos=_fmt(_metric(row, "branch_pressure", "mean_branch_positive_mask")),
                neg=_fmt(_metric(row, "branch_pressure", "mean_branch_negative_mask")),
            )
        )

    lines.extend(
        [
            "",
            "## Major / Residue / Loss",
            "",
            "| variant | major gain | residue drop | loss | l_q_dom | l_q_entropy | l_major_sat |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {major} | {residue} | {loss} | {qdom} | {qentropy} | {major_sat} |".format(
                variant=row.get("variant"),
                major=_fmt(_metric(row, "major_residue_loss", "mean_major_gain")),
                residue=_fmt(_metric(row, "major_residue_loss", "mean_residue_drop")),
                loss=_fmt(_metric(row, "major_residue_loss", "mean_loss")),
                qdom=_fmt(_metric(row, "major_residue_loss", "mean_l_q_dom")),
                qentropy=_fmt(_metric(row, "major_residue_loss", "mean_l_q_entropy")),
                major_sat=_fmt(_metric(row, "major_residue_loss", "mean_l_major_sat")),
            )
        )

    lines.extend(
        [
            "",
            "## Source Split",
            "",
            "| variant | synthetic parent | synthetic child | synthetic phase-only | naked parent | naked child | naked phase-only |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {sp} | {sc} | {sph} | {np} | {nc} | {nph} |".format(
                variant=row.get("variant"),
                sp=_fmt(_source_metric(row, "synthetic", "mean_parent_real_branch_fraction")),
                sc=_fmt(_source_metric(row, "synthetic", "mean_child_real_branch_fraction")),
                sph=_fmt(_source_metric(row, "synthetic", "phase_only_real_branch_fraction")),
                np=_fmt(_source_metric(row, "naked_rafa", "mean_parent_real_branch_fraction")),
                nc=_fmt(_source_metric(row, "naked_rafa", "mean_child_real_branch_fraction")),
                nph=_fmt(_source_metric(row, "naked_rafa", "phase_only_real_branch_fraction")),
            )
        )

    limitations = [str(item) for item in summary.get("limitations", []) if item]
    for row in rows:
        if row.get("error"):
            limitations.append(f"{row.get('variant')}: {row.get('error')}")
        for note in row.get("limitations", []) or []:
            limitations.append(f"{row.get('variant')}: {note}")
    lines.extend(["", "## Limitations", ""])
    if limitations:
        lines.extend(f"- {item}" for item in limitations)
    else:
        lines.append("- No harness-level limitations recorded.")

    (out_dir / SUMMARY_MD_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_substrate_ablation(
    *,
    out_dir: Path,
    base_config_path: Path,
    device_name: str,
    time_steps: int,
    write_only: bool,
    synthetic_only: bool,
    synthetic_seed_start: int,
    synthetic_count: int,
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_payload = _load_json(base_config_path)
    plan = _synthetic_plan(synthetic_seed_start, synthetic_count) if synthetic_only else None
    limitations: list[str] = []
    if synthetic_only and not _can_inject_plan():
        limitations.append("Synthetic-only plan could not be injected; evaluation falls back to evaluate_config.")

    rows: list[dict[str, Any]] = []
    for spec in _variant_specs():
        variant_dir = out_dir / spec["name"]
        config_path = variant_dir / VARIANT_CONFIG_NAME
        payload = _variant_payload(base_payload, spec["overrides"])
        _write_json(config_path, payload)

        heldout = None
        status = "write_only"
        eval_dir = variant_dir / "heldout_eval"
        row_limitations: list[str] = []
        patch_meta: dict[str, Any] = {"requested": spec["smooth_patch"], "applied": False, "limitation": None}

        patch_context: Callable[[], Any]
        if write_only:
            patch_context = nullcontext
        else:
            patch_context = lambda kind=spec["smooth_patch"]: _patched_smoothing(kind, seed)

        try:
            with patch_context() as meta:
                if isinstance(meta, dict):
                    patch_meta = meta
                    if meta.get("limitation"):
                        row_limitations.append(str(meta["limitation"]))
                if not write_only:
                    heldout, plan_limitation = _evaluate_variant(
                        config_path=config_path,
                        eval_dir=eval_dir,
                        time_steps=time_steps,
                        device_name=device_name,
                        plan=plan,
                    )
                    if plan_limitation:
                        row_limitations.append(plan_limitation)
                    status = "evaluated"
        except Exception as exc:
            status = "evaluation_failed"
            heldout = {"error": f"{type(exc).__name__}: {exc}"}

        row = {
            "variant": spec["name"],
            "description": spec["description"],
            "status": status,
            "config_path": str(config_path),
            "heldout_summary": str(eval_dir / "heldout_summary.json") if status == "evaluated" else None,
            "overrides": spec["overrides"],
            "smooth_patch": patch_meta,
            "metrics": _extract_metrics(heldout if status == "evaluated" else None),
            "limitations": row_limitations,
            "error": heldout.get("error") if isinstance(heldout, dict) and "error" in heldout else None,
        }
        rows.append(row)

    summary = {
        "runtime": "circleworld_proto",
        "mode": "substrate_lattice_ablation",
        "base_config_path": str(base_config_path),
        "out_dir": str(out_dir),
        "device": device_name,
        "time_steps": time_steps,
        "write_only": write_only,
        "synthetic_only": synthetic_only,
        "synthetic_plan": plan,
        "seed": seed,
        "limitations": limitations,
        "variants": rows,
    }
    _write_json(out_dir / SUMMARY_JSON_NAME, summary)
    _write_markdown(out_dir, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate and optionally evaluate Circleworld substrate/lattice ablation variants."
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-config", "--config", dest="base_config", default=str(DEFAULT_BASE_CONFIG))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-steps", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260506, help="Deterministic seed for monkeypatched substrate transforms.")
    ap.add_argument("--synthetic-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--synthetic-seed-start", type=int, default=5100)
    ap.add_argument("--synthetic-count", type=int, default=6)
    ap.add_argument("--write-only", action="store_true", help="Only write variant configs and reports without evaluation.")
    args = ap.parse_args()

    summary = run_substrate_ablation(
        out_dir=Path(args.out_dir),
        base_config_path=Path(args.base_config),
        device_name=args.device,
        time_steps=args.time_steps,
        write_only=bool(args.write_only),
        synthetic_only=bool(args.synthetic_only),
        synthetic_seed_start=int(args.synthetic_seed_start),
        synthetic_count=int(args.synthetic_count),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
