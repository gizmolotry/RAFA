from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, RUNTIME, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_circleworld import _load_circle_cfg, evaluate_circle_cfg
from run_child_writeback_ablation import VARIANTS as CHILD_WRITEBACK_VARIANTS
from run_child_writeback_ablation import _jsonable_config as _child_config_payload
from run_q_basis_ablation import _config_block as _q_config_block
from run_q_basis_ablation import _variant_payload as _q_variant_payload
from run_q_basis_ablation import _variant_specs as _q_variant_specs
from run_substrate_ablation import _patched_smoothing, _variant_payload as _substrate_variant_payload
from run_substrate_ablation import _variant_specs as _substrate_variant_specs
import circleworld


DEFAULT_BASE_CONFIG = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "relsig_scout_balance_2026-05-04"
    / "circleworld_real_anchor_config_cem_v1.json"
)

COMMON_METRICS = (
    "count",
    "mean_real_branch_fraction",
    "mean_parent_real_branch_fraction",
    "mean_child_real_branch_fraction",
    "phase_only_real_branch_fraction",
    "phase_only_excess_branch_fraction",
    "mean_decorative_slot2_fraction",
    "decorative_slot2_low_phase_fraction",
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
    "mean_num_law_families",
    "mean_law_family_entropy",
    "mean_dominant_law_q_share",
    "mean_num_relational_signature_families",
    "mean_relational_signature_confidence",
    "mean_major_gain",
    "mean_residue_drop",
    "mean_loss",
    "mean_l_q_dom",
    "mean_l_q_entropy",
    "mean_l_major_sat",
    "final_phase_only_branch_surface_peak",
    "final_phase_only_branch_live_fraction",
    "mean_branch_defect",
    "mean_branch_world_grad",
    "mean_branch_q_disagreement",
    "mean_branch_phase_wall",
)

DELTA_METRICS = (
    "mean_parent_real_branch_fraction",
    "mean_child_real_branch_fraction",
    "phase_only_real_branch_fraction",
    "phase_only_excess_branch_fraction",
    "decorative_slot2_low_phase_fraction",
    "mean_child_writeback_gate_mass",
    "mean_child_phase_writeback_delta_mass",
    "mean_child_support_writeback_mass",
    "mean_child_logit_writeback_mass",
    "mean_child_qtrace_writeback_mass",
    "mean_num_law_families",
    "mean_law_family_entropy",
    "mean_major_gain",
    "mean_loss",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_plan(synthetic_count: int, naked_count: int, synthetic_start: int, naked_start: int) -> list[tuple[str, int]]:
    plan: list[tuple[str, int]] = []
    plan.extend(("synthetic", int(synthetic_start) + i) for i in range(max(0, int(synthetic_count))))
    plan.extend(("naked_rafa", int(naked_start) + i) for i in range(max(0, int(naked_count))))
    if not plan:
        plan.append(("synthetic", int(synthetic_start)))
    return plan


def _pick(source: dict[str, Any] | None, keys: tuple[str, ...] = COMMON_METRICS) -> dict[str, Any]:
    block = source if isinstance(source, dict) else {}
    return {key: block.get(key) for key in keys}


def _summary_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    summary = summary if isinstance(summary, dict) else {}
    by_source = summary.get("by_source")
    by_source = by_source if isinstance(by_source, dict) else {}
    return {
        "top": _pick(summary),
        "by_source": {
            "synthetic": _pick(by_source.get("synthetic")),
            "naked_rafa": _pick(by_source.get("naked_rafa")),
        },
    }


def _numeric(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _deltas(row: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(baseline, dict):
        return {}
    out: dict[str, Any] = {}
    for scope in ("top", "synthetic", "naked_rafa"):
        row_scope = row.get("metrics", {}).get("top" if scope == "top" else "by_source", {})
        base_scope = baseline.get("metrics", {}).get("top" if scope == "top" else "by_source", {})
        if scope != "top":
            row_scope = row_scope.get(scope, {}) if isinstance(row_scope, dict) else {}
            base_scope = base_scope.get(scope, {}) if isinstance(base_scope, dict) else {}
        scope_delta: dict[str, Any] = {}
        for key in DELTA_METRICS:
            lhs = _numeric(row_scope.get(key) if isinstance(row_scope, dict) else None)
            rhs = _numeric(base_scope.get(key) if isinstance(base_scope, dict) else None)
            scope_delta[key] = None if lhs is None or rhs is None else lhs - rhs
        out[scope] = scope_delta
    return out


def _evaluate_config(config_path: Path, out_dir: Path, plan: list[tuple[str, int]], time_steps: int, device: str) -> dict[str, Any]:
    cfg = _load_circle_cfg(config_path)
    return evaluate_circle_cfg(
        cfg,
        time_steps=time_steps,
        device_name=device,
        plan=plan,
        config_path=config_path,
        out_dir=out_dir,
        include_trajectory=True,
    )


def _track_child_writeback(base_config: Path, track_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, overrides in CHILD_WRITEBACK_VARIANTS.items():
        variant_dir = track_dir / name
        config_path = variant_dir / "config.json"
        payload = _child_config_payload(base_config, overrides)
        _write_json(config_path, payload)
        rows.append(
            {
                "variant": name,
                "description": "Child writeback control isolation.",
                "config_path": str(config_path),
                "overrides": overrides,
                "baseline": name == "baseline_current",
                "patch_kind": None,
            }
        )
    return rows


def _track_q_basis(base_payload: dict[str, Any], track_dir: Path, seed: int) -> list[dict[str, Any]]:
    base_cfg = _q_config_block(base_payload)
    rows: list[dict[str, Any]] = []
    for spec in _q_variant_specs(base_cfg, seed):
        name = str(spec["name"])
        variant_dir = track_dir / name
        config_path = variant_dir / "config.json"
        payload = _q_variant_payload(base_payload, spec["overrides"])
        _write_json(config_path, payload)
        rows.append(
            {
                "variant": name,
                "description": spec.get("description"),
                "config_path": str(config_path),
                "overrides": spec.get("overrides"),
                "baseline": name == "ramanujan_baseline",
                "patch_kind": None,
            }
        )
    return rows


def _track_qtrace_inheritance(base_payload: dict[str, Any], track_dir: Path) -> list[dict[str, Any]]:
    specs = [
        {
            "name": "qtrace_baseline",
            "description": "Current q-trace momentum, survival, relation, and child qtrace writeback.",
            "overrides": {},
        },
        {
            "name": "qtrace_current_only_no_memory",
            "description": "Removes q-trace history by replacing the trace with current-window evidence each step.",
            "overrides": {"qtrace_momentum": 0.0},
        },
        {
            "name": "qtrace_frozen_no_write_no_survival",
            "description": "Freezes q-trace and disables qtrace survival/relation influence.",
            "overrides": {
                "qtrace_momentum": 1.0,
                "relation_qtrace_gain": 0.0,
                "survival_qtrace_weight": 0.0,
                "child_survival_qtrace_weight": 0.0,
            },
        },
        {
            "name": "qtrace_survival_weights_off",
            "description": "Stores q-trace but removes qtrace terms from parent and child survival.",
            "overrides": {
                "survival_qtrace_weight": 0.0,
                "child_survival_qtrace_weight": 0.0,
            },
        },
        {
            "name": "qtrace_relation_gain_off",
            "description": "Keeps q-trace memory/survival but removes relational qtrace residual drive.",
            "overrides": {"relation_qtrace_gain": 0.0},
        },
        {
            "name": "qtrace_child_survival_off",
            "description": "Keeps parent qtrace survival but removes child-local qtrace survival weight.",
            "overrides": {"child_survival_qtrace_weight": 0.0},
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec["name"])
        variant_dir = track_dir / name
        config_path = variant_dir / "config.json"
        payload = _q_variant_payload(base_payload, spec["overrides"])
        _write_json(config_path, payload)
        rows.append(
            {
                "variant": name,
                "description": spec["description"],
                "config_path": str(config_path),
                "overrides": spec["overrides"],
                "baseline": name == "qtrace_baseline",
                "patch_kind": None,
            }
        )
    return rows


def _track_substrate(base_payload: dict[str, Any], track_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _substrate_variant_specs():
        name = str(spec["name"])
        variant_dir = track_dir / name
        config_path = variant_dir / "config.json"
        payload = _substrate_variant_payload(base_payload, spec["overrides"])
        _write_json(config_path, payload)
        rows.append(
            {
                "variant": name,
                "description": spec.get("description"),
                "config_path": str(config_path),
                "overrides": spec.get("overrides"),
                "baseline": name == "lattice_baseline",
                "patch_kind": spec.get("smooth_patch"),
            }
        )
    return rows


def _track_arc_lane(base_payload: dict[str, Any], track_dir: Path) -> list[dict[str, Any]]:
    specs = [
        {
            "name": "arc_baseline",
            "description": "Unmodified Hardy-Littlewood arc and promotability fields.",
            "patch_kind": None,
        },
        {
            "name": "arc_flat_major_residue",
            "description": "Flattens major/minor/concentration/sharpness over time while preserving per-q trace.",
            "patch_kind": "flat_major_residue",
        },
        {
            "name": "arc_time_shuffled",
            "description": "Deterministically shuffles time-aligned arc fields, preserving marginal arc distributions.",
            "patch_kind": "time_shuffled",
        },
        {
            "name": "arc_major_off_residue_high",
            "description": "Removes major-arc support and sets residue high while preserving per-q trace.",
            "patch_kind": "major_off_residue_high",
        },
        {
            "name": "promotability_flat",
            "description": "Keeps arc fields but flattens promotability/persistence over time.",
            "patch_kind": "promotability_flat",
        },
        {
            "name": "promotability_zero",
            "description": "Keeps arc fields but disables packet promotion by zeroing promotability/persistence.",
            "patch_kind": "promotability_zero",
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec["name"])
        variant_dir = track_dir / name
        config_path = variant_dir / "config.json"
        _write_json(config_path, base_payload)
        rows.append(
            {
                "variant": name,
                "description": spec["description"],
                "config_path": str(config_path),
                "overrides": {},
                "baseline": name == "arc_baseline",
                "patch_kind": spec.get("patch_kind"),
            }
        )
    return rows


def _track_arc_q_cross(base_payload: dict[str, Any], track_dir: Path, seed: int) -> list[dict[str, Any]]:
    base_cfg = _q_config_block(base_payload)
    q_specs_by_name = {str(spec["name"]): spec for spec in _q_variant_specs(base_cfg, seed)}
    q_names = [
        "ramanujan_baseline",
        "qtrace_only_control",
        "uniform_q_weights",
        "shuffled_qset",
        "compact_fourierish_control",
        "compact_qtrace_only_control",
        "phase_only_relation_control",
        "uniform_relation_kernel_control",
    ]
    arc_patches = [
        ("arc_baseline", None),
        ("arc_flat_major_residue", "flat_major_residue"),
        ("arc_time_shuffled", "time_shuffled"),
        ("promotability_zero", "promotability_zero"),
    ]
    rows: list[dict[str, Any]] = []
    for q_name in q_names:
        q_spec = q_specs_by_name[q_name]
        for arc_name, patch_kind in arc_patches:
            name = f"{q_name}__{arc_name}"
            variant_dir = track_dir / name
            config_path = variant_dir / "config.json"
            payload = _q_variant_payload(base_payload, q_spec["overrides"])
            _write_json(config_path, payload)
            rows.append(
                {
                    "variant": name,
                    "description": f"Crosses q control `{q_name}` with arc patch `{arc_name}`.",
                    "config_path": str(config_path),
                    "overrides": q_spec.get("overrides"),
                    "q_variant": q_name,
                    "arc_variant": arc_name,
                    "baseline": name == "ramanujan_baseline__arc_baseline",
                    "patch_kind": patch_kind,
                }
            )
    return rows


def _deterministic_time_perm(size: int, seed: int, device: torch.device) -> torch.Tensor:
    if size <= 1:
        return torch.arange(size, device=device, dtype=torch.long)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randperm(size, device=device, generator=gen)


def _shuffle_last_dim(value: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    if value.dim() == 0 or value.size(-1) != perm.numel():
        return value
    return value.index_select(-1, perm)


@contextmanager
def _patched_arc_lane(kind: str | None, seed: int) -> Iterator[dict[str, Any]]:
    original_arc = circleworld.hardy_littlewood_arc_field
    original_promo = circleworld.promotability_field
    meta = {"requested": kind, "applied": False, "limitation": None}
    if kind is None:
        yield meta
        return

    def patched_arc(z: torch.Tensor, qset: tuple[int, ...], q_weights: tuple[float, ...]) -> dict[str, torch.Tensor]:
        arc = original_arc(z, qset, q_weights)
        if kind == "flat_major_residue":
            for key in ("major_mass", "minor_residue", "harmonic_ratio", "concentration", "sharpness"):
                field = arc.get(key)
                if torch.is_tensor(field) and field.dim() >= 1:
                    arc[key] = field.mean(dim=-1, keepdim=True).expand_as(field).clone()
        elif kind == "time_shuffled":
            t_len = z.size(2)
            perm = _deterministic_time_perm(t_len, seed, z.device)
            for key, field in list(arc.items()):
                if torch.is_tensor(field):
                    arc[key] = _shuffle_last_dim(field, perm)
        elif kind == "major_off_residue_high":
            for key in ("major_mass", "harmonic_ratio", "concentration", "sharpness"):
                field = arc.get(key)
                if torch.is_tensor(field):
                    arc[key] = torch.zeros_like(field)
            residue = arc.get("minor_residue")
            if torch.is_tensor(residue):
                arc["minor_residue"] = torch.ones_like(residue)
        return arc

    def patched_promotability(arc: dict[str, torch.Tensor], persistence_momentum: float = 0.6) -> dict[str, torch.Tensor]:
        promo = original_promo(arc, persistence_momentum=persistence_momentum)
        if kind == "promotability_flat":
            for key, field in list(promo.items()):
                if torch.is_tensor(field) and field.dim() >= 1:
                    promo[key] = field.mean(dim=-1, keepdim=True).expand_as(field).clone()
        elif kind == "promotability_zero":
            for key, field in list(promo.items()):
                if torch.is_tensor(field):
                    promo[key] = torch.zeros_like(field)
        return promo

    if kind in {"flat_major_residue", "time_shuffled", "major_off_residue_high"}:
        circleworld.hardy_littlewood_arc_field = patched_arc
    elif kind in {"promotability_flat", "promotability_zero"}:
        circleworld.promotability_field = patched_promotability
    else:
        meta["limitation"] = f"Unknown arc patch kind: {kind}"
        yield meta
        return
    meta["applied"] = True
    try:
        yield meta
    finally:
        circleworld.hardy_littlewood_arc_field = original_arc
        circleworld.promotability_field = original_promo


def _patch_context(track: str, patch_kind: str | None, seed: int) -> Iterator[dict[str, Any] | None]:
    if track == "substrate" and patch_kind is not None:
        return _patched_smoothing(patch_kind, seed)
    if track in {"arc_lane", "arc_q_cross"} and patch_kind is not None:
        return _patched_arc_lane(patch_kind, seed)
    if patch_kind is None:
        return nullcontext(None)
    return nullcontext({"requested": patch_kind, "applied": False, "limitation": f"No patch context for track {track}."})


def _run_track(
    *,
    track: str,
    rows: list[dict[str, Any]],
    out_dir: Path,
    plan: list[tuple[str, int]],
    time_steps: int,
    device: str,
    write_only: bool,
    seed: int,
) -> dict[str, Any]:
    evaluated_rows: list[dict[str, Any]] = []
    for row in rows:
        variant_dir = out_dir / track / row["variant"]
        eval_dir = variant_dir / "heldout_eval"
        status = "write_only"
        patch_meta: dict[str, Any] | None = None
        if not write_only:
            try:
                with _patch_context(track, row.get("patch_kind"), seed) as meta:
                    patch_meta = meta if isinstance(meta, dict) else None
                    summary = _evaluate_config(
                        config_path=Path(row["config_path"]),
                        out_dir=eval_dir,
                        plan=plan,
                        time_steps=time_steps,
                        device=device,
                    )
                row["heldout_summary"] = str(eval_dir / "heldout_summary.json")
                row["metrics"] = _summary_metrics(summary)
                status = "evaluated"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                status = "failed"
        row["status"] = status
        if patch_meta is not None:
            row["patch"] = patch_meta
        evaluated_rows.append(row)

    baseline = next((row for row in evaluated_rows if row.get("baseline")), None)
    for row in evaluated_rows:
        row["delta_vs_track_baseline"] = _deltas(row, baseline)
    return {
        "track": track,
        "baseline_variant": baseline.get("variant") if isinstance(baseline, dict) else None,
        "variants": evaluated_rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _metric(row: dict[str, Any], key: str, source: str = "top") -> Any:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    if source == "top":
        block = metrics.get("top")
    else:
        by_source = metrics.get("by_source")
        block = by_source.get(source) if isinstance(by_source, dict) else None
    return block.get(key) if isinstance(block, dict) else None


def _delta(row: dict[str, Any], key: str, source: str = "top") -> Any:
    deltas = row.get("delta_vs_track_baseline")
    if not isinstance(deltas, dict):
        return None
    block = deltas.get(source)
    return block.get(key) if isinstance(block, dict) else None


def _write_markdown(out_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# RAFA Claim Isolation Suite",
        "",
        f"- base config: `{report.get('base_config_path')}`",
        f"- device: `{report.get('device')}`",
        f"- time steps: `{report.get('time_steps')}`",
        f"- write only: `{report.get('write_only')}`",
        f"- seed plan: `{report.get('seed_plan')}`",
        "",
    ]
    for track in report.get("tracks", []):
        lines.extend(
            [
                f"## {track.get('track')}",
                "",
                "| variant | status | parent | child | phase-only | naked phase | gate | phase delta | support | law fam | major gain | loss | d loss |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in track.get("variants", []):
            lines.append(
                "| {variant} | {status} | {parent} | {child} | {phase} | {naked_phase} | {gate} | {phase_delta} | {support} | {law} | {major} | {loss} | {dloss} |".format(
                    variant=row.get("variant"),
                    status=row.get("status"),
                    parent=_fmt(_metric(row, "mean_parent_real_branch_fraction")),
                    child=_fmt(_metric(row, "mean_child_real_branch_fraction")),
                    phase=_fmt(_metric(row, "phase_only_real_branch_fraction")),
                    naked_phase=_fmt(_metric(row, "phase_only_real_branch_fraction", "naked_rafa")),
                    gate=_fmt(_metric(row, "mean_child_writeback_gate_mass")),
                    phase_delta=_fmt(_metric(row, "mean_child_phase_writeback_delta_mass")),
                    support=_fmt(_metric(row, "mean_child_support_writeback_mass")),
                    law=_fmt(_metric(row, "mean_num_law_families")),
                    major=_fmt(_metric(row, "mean_major_gain")),
                    loss=_fmt(_metric(row, "mean_loss")),
                    dloss=_fmt(_delta(row, "mean_loss")),
                )
            )
        lines.append("")
    (out_dir / "CLAIM_ISOLATION_SUITE.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite(
    *,
    base_config_path: Path,
    out_dir: Path,
    tracks: list[str],
    plan: list[tuple[str, int]],
    time_steps: int,
    device: str,
    write_only: bool,
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_payload = _load_json(base_config_path)
    track_rows: dict[str, list[dict[str, Any]]] = {
        "arc_lane": _track_arc_lane(base_payload, out_dir / "arc_lane"),
        "arc_q_cross": _track_arc_q_cross(base_payload, out_dir / "arc_q_cross", seed),
        "child_writeback": _track_child_writeback(base_config_path, out_dir / "child_writeback"),
        "q_basis": _track_q_basis(base_payload, out_dir / "q_basis", seed),
        "qtrace_inheritance": _track_qtrace_inheritance(base_payload, out_dir / "qtrace_inheritance"),
        "substrate": _track_substrate(base_payload, out_dir / "substrate"),
    }
    selected_tracks = [track for track in tracks if track in track_rows]
    report = {
        "runtime": "circleworld_proto",
        "schema": "rafa_claim_isolation_suite_v0",
        "base_config_path": str(base_config_path),
        "out_dir": str(out_dir),
        "device": device,
        "time_steps": time_steps,
        "write_only": write_only,
        "seed": seed,
        "seed_plan": [{"source": source, "seed": seed_value} for source, seed_value in plan],
        "tracks": [
            _run_track(
                track=track,
                rows=track_rows[track],
                out_dir=out_dir,
                plan=plan,
                time_steps=time_steps,
                device=device,
                write_only=write_only,
                seed=seed,
            )
            for track in selected_tracks
        ],
    }
    _write_json(out_dir / "claim_isolation_suite_summary.json", report)
    _write_markdown(out_dir, report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run paired RAFA claim-isolation ablations on one fixed seed plan.")
    ap.add_argument("--base-config", "--config", dest="base_config", default=str(DEFAULT_BASE_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-steps", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260506)
    ap.add_argument("--synthetic-count", type=int, default=3)
    ap.add_argument("--naked-count", type=int, default=3)
    ap.add_argument("--synthetic-seed-start", type=int, default=5100)
    ap.add_argument("--naked-seed-start", type=int, default=6100)
    ap.add_argument(
        "--tracks",
        nargs="+",
        default=["child_writeback", "q_basis", "substrate"],
        choices=["arc_lane", "arc_q_cross", "child_writeback", "q_basis", "qtrace_inheritance", "substrate"],
    )
    ap.add_argument("--write-only", action="store_true")
    args = ap.parse_args()

    plan = _seed_plan(
        synthetic_count=int(args.synthetic_count),
        naked_count=int(args.naked_count),
        synthetic_start=int(args.synthetic_seed_start),
        naked_start=int(args.naked_seed_start),
    )
    report = run_suite(
        base_config_path=Path(args.base_config),
        out_dir=Path(args.out_dir),
        tracks=list(args.tracks),
        plan=plan,
        time_steps=int(args.time_steps),
        device=str(args.device),
        write_only=bool(args.write_only),
        seed=int(args.seed),
    )
    print(json.dumps({"summary": str(Path(args.out_dir) / "claim_isolation_suite_summary.json"), "tracks": args.tracks}, indent=2))


if __name__ == "__main__":
    main()
