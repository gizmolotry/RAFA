from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))



def _exit_if_help_requested_without_runtime() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(f"usage: {Path(__file__).name} [runtime-coupled contract options]")
        print()
        print("This contract is coupled to the Circleworld runtime/signature lane.")
        print("Run it from a Circleworld integration worktree for full argument parsing and execution.")
        raise SystemExit(0)


_exit_if_help_requested_without_runtime()

from ablate_formalization import DEFAULT_SEED_RAFA_INIT, _generate_seed_phase, make_seed_rafa
from evaluate_circleworld import _heldout_plan, _load_circle_cfg, evaluate_circle_cfg

DEFAULT_CONTROLLER_MANIFEST = ROOT / "outputs" / "circleworld_proto" / "tokenburst_2026-05-04_controller" / "baseline_manifest.json"
DEFAULT_BRANCH_FIRST = ROOT / "checkpoints_circleworld_proto" / "training_run_2026-04-24_agreement_scout_v1" / "circleworld_real_anchor_config_cem_v1.json"
DEFAULT_DIVERSITY_REFERENCE = ROOT / "checkpoints_circleworld_proto" / "manual_candidates" / "circleworld_parentmix_220_100_2026-04-24.json"
DEFAULT_SIGNATURE_INIT = ROOT / "checkpoints_circleworld_proto" / "relsig_scout_balance_2026-05-04" / "circleworld_real_anchor_config_cem_v1.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "circleworld_proto" / "tokenburst_2026-05-04_w7_seed_determinism"

SUMMARY_KEYS = (
    "mean_real_branch_fraction",
    "mean_meso_branch_effect",
    "mean_live_child_fraction",
    "mean_child_writeback_mass",
    "mean_child_parent_divergence",
    "mean_num_law_families",
    "mean_law_family_entropy",
)
ROW_METRIC_KEYS = (
    "real_branch_fraction",
    "meso_branch_effect",
    "mean_live_child_fraction",
    "mean_child_writeback_mass",
    "mean_child_parent_divergence",
    "num_law_families",
    "law_family_entropy",
)
ROW_FLAG_KEY_ORDER = ("branch_active", "selection_ready")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(payload: Any) -> str:
    return _sha256_text(_stable_json(payload))


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return _sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _rng_digest(device: torch.device) -> dict[str, str | None]:
    digest = {
        "python_random": _sha256_bytes(pickle.dumps(random.getstate(), protocol=pickle.HIGHEST_PROTOCOL)),
        "torch_cpu": _sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
        "torch_cuda": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_bytes = b"".join(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
        digest["torch_cuda"] = _sha256_bytes(cuda_bytes)
    return digest


def _perturb_global_rng(tag: int, device: torch.device) -> dict[str, str | None]:
    random.seed(11003 + tag)
    torch.manual_seed(22007 + tag)
    _ = torch.randn(19)
    _ = torch.rand(11)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(33013 + tag)
        _ = torch.randn((23,), device=device)
        _ = torch.rand((17,), device=device)
    return _rng_digest(device)


def _normalize_plan(raw_plan: list[Any]) -> list[tuple[str, int]]:
    plan: list[tuple[str, int]] = []
    for source, seed in raw_plan:
        plan.append((str(source), int(seed)))
    return plan


def _plan_summary(plan: list[tuple[str, int]]) -> dict[str, Any]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for source, seed in plan:
        by_source[source].append(int(seed))
    families = {
        source: {
            "count": len(seeds),
            "seeds": sorted(seeds),
        }
        for source, seeds in sorted(by_source.items())
    }
    return {
        "count": len(plan),
        "families": families,
        "sha256": _sha256_json(plan),
    }


def _selection_targets(probe_cfg: dict[str, Any], source: str) -> dict[str, float]:
    targets = {
        "real_branch_fraction": float(
            probe_cfg.get("min_naked_branch_target", probe_cfg.get("min_branch_target", 0.0))
            if source == "naked_rafa"
            else probe_cfg.get("min_branch_target", 0.0)
        ),
        "mean_live_child_fraction": float(
            probe_cfg.get("min_naked_live_child_target", probe_cfg.get("min_live_child_target", 0.0))
            if source == "naked_rafa"
            else probe_cfg.get("min_live_child_target", 0.0)
        ),
        "mean_child_writeback_mass": float(
            probe_cfg.get("min_naked_writeback_target", probe_cfg.get("min_writeback_target", 0.0))
            if source == "naked_rafa"
            else probe_cfg.get("min_writeback_target", 0.0)
        ),
        "mean_child_parent_divergence": float(
            probe_cfg.get("min_naked_parent_div_target", probe_cfg.get("min_parent_div_target", 0.0))
            if source == "naked_rafa"
            else probe_cfg.get("min_parent_div_target", 0.0)
        ),
        "num_law_families": float(
            probe_cfg.get("min_naked_law_family_target", probe_cfg.get("min_law_family_target", 0.0))
            if source == "naked_rafa"
            else probe_cfg.get("min_law_family_target", 0.0)
        ),
        "law_family_entropy": float(
            probe_cfg.get("min_naked_law_entropy_target", probe_cfg.get("min_law_entropy_target", 0.0))
            if source == "naked_rafa"
            else probe_cfg.get("min_law_entropy_target", 0.0)
        ),
    }
    return {name: value for name, value in targets.items() if value > 0.0}


def _selection_targets_enabled(probe_cfg: dict[str, Any]) -> bool:
    return any(float(value) > 0.0 for source in ("synthetic", "naked_rafa") for value in _selection_targets(probe_cfg, source).values())


def _row_id(row: dict[str, Any]) -> str:
    return f"{row['source']}::{int(row['seed'])}"


def _row_flags(row: dict[str, Any], probe_cfg: dict[str, Any]) -> dict[str, Any]:
    branch_active = float(row.get("real_branch_fraction", 0.0)) > 0.0
    targets = _selection_targets(probe_cfg, str(row["source"]))
    checks: dict[str, dict[str, float | bool]] = {}
    selection_ready: bool | None = None
    if targets:
        selection_ready = True
        for metric, target in targets.items():
            value = float(row.get(metric, 0.0))
            passed = bool(value >= target)
            checks[metric] = {
                "value": value,
                "target": target,
                "passed": passed,
            }
            selection_ready = bool(selection_ready and passed)
    return {
        "branch_active": branch_active,
        "selection_ready": selection_ready,
        "checks": checks,
    }


def _summary_slice(summary: dict[str, Any]) -> dict[str, Any]:
    payload = {key: float(summary.get(key, 0.0)) for key in SUMMARY_KEYS}
    payload["by_source"] = {}
    for source, source_summary in sorted(summary.get("by_source", {}).items()):
        payload["by_source"][source] = {
            key: float(source_summary.get(key, 0.0))
            for key in (
                "mean_real_branch_fraction",
                "mean_meso_branch_effect",
                "mean_live_child_fraction",
                "mean_child_writeback_mass",
                "mean_child_parent_divergence",
                "mean_num_law_families",
                "mean_law_family_entropy",
            )
        }
    return payload


def _row_metric_spreads(repeat_rows: list[list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    metric_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rows in repeat_rows:
        for row in rows:
            row_id = _row_id(row)
            for metric in ROW_METRIC_KEYS:
                metric_values[(row_id, metric)].append(float(row.get(metric, 0.0)))

    spreads: dict[str, dict[str, Any]] = {}
    for metric in ROW_METRIC_KEYS:
        best_row_id = None
        best_spread = -1.0
        for (row_id, metric_name), values in metric_values.items():
            if metric_name != metric:
                continue
            spread = float(max(values) - min(values))
            if spread > best_spread:
                best_row_id = row_id
                best_spread = spread
        spreads[metric] = {
            "max_spread": float(max(0.0, best_spread)),
            "row_id": best_row_id,
        }
    return spreads


def _phase_seed_check(
    *,
    device: torch.device,
    seed_source: str,
    seed: int,
    time_steps: int,
    repeats: int,
    rafa_init_seed: int,
) -> dict[str, Any]:
    phase_hashes: list[str] = []
    rng_preserved: list[bool] = []
    max_abs_delta = 0.0
    anchor: torch.Tensor | None = None

    for rep in range(repeats):
        before = _perturb_global_rng((seed * 17) + rep + (0 if seed_source == "synthetic" else 100000), device)
        rafa_core = make_seed_rafa(dev=device.type, init_seed=rafa_init_seed) if seed_source == "naked_rafa" else None
        phase_state = _generate_seed_phase(
            rafa_core=rafa_core,
            batch_size=1,
            time_steps=time_steps,
            device=device,
            seed_source=seed_source,
            seed=seed,
        ).detach()
        after = _rng_digest(device)
        phase_hashes.append(_sha256_tensor(phase_state))
        rng_preserved.append(after == before)

        phase_cpu = phase_state.cpu()
        if anchor is None:
            anchor = phase_cpu
        else:
            max_abs_delta = max(max_abs_delta, float((phase_cpu - anchor).abs().max().item()))

    return {
        "source": seed_source,
        "seed": int(seed),
        "time_steps": int(time_steps),
        "repeats": int(repeats),
        "phase_sha256": phase_hashes,
        "exact_match": len(set(phase_hashes)) == 1,
        "all_rng_preserved": all(rng_preserved),
        "rng_preserved_repeats": rng_preserved,
        "max_abs_delta_vs_repeat0": float(max_abs_delta),
    }


def _audit_phase_union(
    *,
    device: torch.device,
    configs: list[dict[str, Any]],
    repeats: int,
    rafa_init_seed: int,
) -> dict[str, Any]:
    phase_targets = sorted({(source, int(seed), int(cfg["time_steps"])) for cfg in configs for source, seed in cfg["plan"]})
    checks = [
        _phase_seed_check(
            device=device,
            seed_source=source,
            seed=seed,
            time_steps=time_steps,
            repeats=repeats,
            rafa_init_seed=rafa_init_seed,
        )
        for source, seed, time_steps in phase_targets
    ]

    by_family: dict[str, dict[str, Any]] = {}
    for source in sorted({item["source"] for item in checks}):
        family_checks = [item for item in checks if item["source"] == source]
        by_family[source] = {
            "count": len(family_checks),
            "seeds": [int(item["seed"]) for item in family_checks],
            "all_exact_match": all(bool(item["exact_match"]) for item in family_checks),
            "all_rng_preserved": all(bool(item["all_rng_preserved"]) for item in family_checks),
            "max_abs_delta_vs_repeat0": float(max((item["max_abs_delta_vs_repeat0"] for item in family_checks), default=0.0)),
            "checks": family_checks,
        }

    return {
        "repeats": int(repeats),
        "rafa_init_seed": int(rafa_init_seed),
        "families": by_family,
    }


def _resolve_config_record(path: Path, label: str, device: torch.device) -> dict[str, Any]:
    payload = _read_json(path)
    probe_cfg = dict(payload.get("heldout_probe_cfg", {}) or {})
    raw_plan = probe_cfg.get("plan")
    if raw_plan:
        plan = _normalize_plan(list(raw_plan))
        plan_source = "config.heldout_probe_cfg.plan"
    else:
        plan = _heldout_plan(device)
        plan_source = "evaluate_circleworld._heldout_plan"
    return {
        "label": label,
        "config_path": str(path),
        "config_seed": payload.get("seed"),
        "requested_device": str(payload.get("device", "cuda")),
        "time_steps": int(probe_cfg.get("time_steps", 128)),
        "selection_probe_repeats": int(probe_cfg.get("selection_probe_repeats", 1)),
        "probe_cfg": probe_cfg,
        "selection_targets_enabled": _selection_targets_enabled(probe_cfg),
        "plan": plan,
        "plan_source": plan_source,
        "plan_summary": _plan_summary(plan),
    }


def _evaluation_repeat_audit(
    *,
    cfg_path: Path,
    device: torch.device,
    plan: list[tuple[str, int]],
    time_steps: int,
    repeats: int,
    probe_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg = _load_circle_cfg(cfg_path)
    repeat_rows: list[list[dict[str, Any]]] = []
    repeat_hashes: list[str] = []
    rng_preserved: list[bool] = []
    flag_maps: list[dict[str, dict[str, Any]]] = []
    summary_sample: dict[str, Any] | None = None

    for rep in range(repeats):
        before = _perturb_global_rng(500000 + rep, device)
        summary = evaluate_circle_cfg(
            cfg,
            time_steps=time_steps,
            device_name=device.type,
            plan=plan,
            include_trajectory=False,
        )
        after = _rng_digest(device)
        rows = list(summary["rows"])
        repeat_rows.append(rows)
        repeat_hashes.append(_sha256_json(rows))
        rng_preserved.append(after == before)
        flag_maps.append({_row_id(row): _row_flags(row, probe_cfg) for row in rows})
        if summary_sample is None:
            summary_sample = _summary_slice(summary)

    assert summary_sample is not None

    branch_flag_stable = True
    selection_flag_stable = True
    if flag_maps:
        first = flag_maps[0]
        for other in flag_maps[1:]:
            for row_id in first.keys():
                if first[row_id]["branch_active"] != other[row_id]["branch_active"]:
                    branch_flag_stable = False
                if first[row_id]["selection_ready"] != other[row_id]["selection_ready"]:
                    selection_flag_stable = False

    family_summary: dict[str, dict[str, Any]] = {}
    first_flags = flag_maps[0] if flag_maps else {}
    first_rows = repeat_rows[0] if repeat_rows else []
    for source in sorted({str(row["source"]) for row in first_rows}):
        row_ids = [_row_id(row) for row in first_rows if str(row["source"]) == source]
        branch_active_count = sum(1 for row_id in row_ids if bool(first_flags[row_id]["branch_active"]))
        selection_ready_values = [first_flags[row_id]["selection_ready"] for row_id in row_ids]
        selection_ready_count = sum(1 for value in selection_ready_values if value is True)
        family_summary[source] = {
            "count": len(row_ids),
            "branch_active_case_count": int(branch_active_count),
            "selection_ready_case_count": int(selection_ready_count),
            "selection_ready_enabled": any(value is not None for value in selection_ready_values),
            "rows": [
                {
                    "row_id": row_id,
                    **first_flags[row_id],
                }
                for row_id in row_ids
            ],
        }

    return {
        "num_repeats": int(repeats),
        "rows_sha256": repeat_hashes,
        "exact_rows_match": len(set(repeat_hashes)) == 1,
        "all_rng_preserved": all(rng_preserved),
        "rng_preserved_repeats": rng_preserved,
        "branch_active_flags_stable": branch_flag_stable,
        "selection_ready_flags_stable": selection_flag_stable,
        "row_metric_spreads": _row_metric_spreads(repeat_rows),
        "summary_sample": summary_sample,
        "family_summary": family_summary,
        "rows": first_rows,
    }


def _pairwise_compare(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    lhs_eval = lhs["evaluation"]
    rhs_eval = rhs["evaluation"]
    lhs_rows = {entry["row_id"]: entry for source in lhs_eval["family_summary"].values() for entry in source["rows"]}
    rhs_rows = {entry["row_id"]: entry for source in rhs_eval["family_summary"].values() for entry in source["rows"]}
    shared_row_ids = sorted(set(lhs_rows).intersection(rhs_rows))

    only_lhs_branch = [row_id for row_id in shared_row_ids if bool(lhs_rows[row_id]["branch_active"]) and not bool(rhs_rows[row_id]["branch_active"])]
    only_rhs_branch = [row_id for row_id in shared_row_ids if bool(rhs_rows[row_id]["branch_active"]) and not bool(lhs_rows[row_id]["branch_active"])]
    only_lhs_ready = [
        row_id
        for row_id in shared_row_ids
        if lhs_rows[row_id]["selection_ready"] is True and rhs_rows[row_id]["selection_ready"] is not True
    ]
    only_rhs_ready = [
        row_id
        for row_id in shared_row_ids
        if rhs_rows[row_id]["selection_ready"] is True and lhs_rows[row_id]["selection_ready"] is not True
    ]

    overall_deltas = {
        key: float(rhs_eval["summary_sample"].get(key, 0.0) - lhs_eval["summary_sample"].get(key, 0.0))
        for key in SUMMARY_KEYS
    }
    by_source_deltas: dict[str, dict[str, float]] = {}
    lhs_by_source = lhs_eval["summary_sample"].get("by_source", {})
    rhs_by_source = rhs_eval["summary_sample"].get("by_source", {})
    for source in sorted(set(lhs_by_source).intersection(rhs_by_source)):
        by_source_deltas[source] = {
            key: float(rhs_by_source[source].get(key, 0.0) - lhs_by_source[source].get(key, 0.0))
            for key in rhs_by_source[source].keys()
        }

    return {
        "lhs": lhs["label"],
        "rhs": rhs["label"],
        "same_plan": lhs["plan_summary"]["sha256"] == rhs["plan_summary"]["sha256"],
        "lhs_plan_source": lhs["plan_source"],
        "rhs_plan_source": rhs["plan_source"],
        "lhs_time_steps": int(lhs["time_steps"]),
        "rhs_time_steps": int(rhs["time_steps"]),
        "overall_deltas_rhs_minus_lhs": overall_deltas,
        "by_source_deltas_rhs_minus_lhs": by_source_deltas,
        "branch_active_only_lhs": only_lhs_branch,
        "branch_active_only_rhs": only_rhs_branch,
        "selection_ready_only_lhs": only_lhs_ready,
        "selection_ready_only_rhs": only_rhs_ready,
    }


def _build_compare(summary: dict[str, Any]) -> dict[str, Any]:
    configs = list(summary["configs"])
    pairwise = []
    for idx, lhs in enumerate(configs):
        for rhs in configs[idx + 1 :]:
            pairwise.append(_pairwise_compare(lhs, rhs))
    return {
        "generated_utc": summary["generated_utc"],
        "device_used": summary["device_used"],
        "phase_union_sha256": _sha256_json(summary["phase_seed_audit"]),
        "pairwise": pairwise,
    }


def _build_report(summary: dict[str, Any], compare: dict[str, Any]) -> str:
    lines = [
        "# W7 Seed Determinism Audit",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Device used: `{summary['device_used']}`",
        f"- Controller manifest: `{summary['controller_manifest']}`",
        "",
        "## Seed Paths",
    ]
    for cfg in summary["configs"]:
        lines.append(
            f"- `{cfg['label']}`: `{cfg['plan_source']}`, `time_steps={cfg['time_steps']}`, "
            f"`selection_probe_repeats={cfg['selection_probe_repeats']}`, plan sha256 `{cfg['plan_summary']['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Exact Reproducibility Checks",
            f"1. `_generate_seed_phase()` SHA-256 exact match across `{summary['phase_seed_audit']['repeats']}` repeats for every audited `(seed_source, seed, time_steps)` tuple after deliberate global RNG perturbation.",
            "2. Global RNG digest preservation across each `_generate_seed_phase()` call.",
            "3. `evaluate_circle_cfg(..., include_trajectory=False)` heldout `rows` SHA-256 exact match across repeated full-plan evaluations for each audited config after deliberate global RNG perturbation.",
            "4. Global RNG digest preservation across each repeated full-plan evaluation.",
            "5. Per-seed `branch_active` flag stability across repeated full-plan evaluations.",
            "6. Per-seed `selection_ready` flag stability across repeated full-plan evaluations when the config provides heldout thresholds.",
            "",
            "## Results",
        ]
    )

    for source, payload in summary["phase_seed_audit"]["families"].items():
        lines.append(
            f"- Phase family `{source}`: `all_exact_match={payload['all_exact_match']}`, "
            f"`all_rng_preserved={payload['all_rng_preserved']}`, `max_abs_delta_vs_repeat0={payload['max_abs_delta_vs_repeat0']}`"
        )
    for cfg in summary["configs"]:
        eval_payload = cfg["evaluation"]
        lines.append(
            f"- Eval `{cfg['label']}`: `exact_rows_match={eval_payload['exact_rows_match']}`, "
            f"`all_rng_preserved={eval_payload['all_rng_preserved']}`, "
            f"`branch_active_flags_stable={eval_payload['branch_active_flags_stable']}`, "
            f"`selection_ready_flags_stable={eval_payload['selection_ready_flags_stable']}`"
        )

    lines.extend(["", "## Pairwise Comparison"])
    for row in compare["pairwise"]:
        lines.append(
            f"- `{row['lhs']}` vs `{row['rhs']}`: `same_plan={row['same_plan']}`, "
            f"`branch_active_only_lhs={len(row['branch_active_only_lhs'])}`, "
            f"`branch_active_only_rhs={len(row['branch_active_only_rhs'])}`"
        )

    lines.extend(
        [
            "",
            "## Risks",
            "- The audit covers the explicit heldout seed plans and their evaluator path; unseen seed families or alternate time-step schedules remain outside this pass.",
            "- Results are tied to the current CUDA stack and single visible GPU. Cross-machine or cross-driver determinism is not established here.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(
    *,
    controller_manifest: Path,
    branch_first: Path,
    diversity_reference: Path,
    signature_init: Path,
    out_dir: Path,
    device_name: str,
    repeats: int,
    phase_repeats: int,
    rafa_init_seed: int,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    baseline_manifest = _read_json(controller_manifest)
    configs = [
        _resolve_config_record(branch_first, "branch_first_active", device),
        _resolve_config_record(diversity_reference, "diversity_reference", device),
        _resolve_config_record(signature_init, "signature_init", device),
    ]
    phase_seed_audit = _audit_phase_union(
        device=device,
        configs=configs,
        repeats=phase_repeats,
        rafa_init_seed=rafa_init_seed,
    )
    for cfg in configs:
        cfg["evaluation"] = _evaluation_repeat_audit(
            cfg_path=Path(cfg["config_path"]),
            device=device,
            plan=list(cfg["plan"]),
            time_steps=int(cfg["time_steps"]),
            repeats=repeats,
            probe_cfg=dict(cfg["probe_cfg"]),
        )

    summary = {
        "generated_utc": _utc_now(),
        "scope": "circleworld_only",
        "controller_manifest": str(controller_manifest),
        "baseline_manifest": baseline_manifest,
        "device_used": device.type,
        "phase_seed_audit": phase_seed_audit,
        "configs": configs,
    }
    compare = _build_compare(summary)
    report = _build_report(summary, compare)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "track_summary.json", summary)
    _write_json(out_dir / "track_compare.json", compare)
    (out_dir / "TRACK_REPORT.md").write_text(report, encoding="utf-8")
    return {
        "summary": summary,
        "compare": compare,
        "report_path": str(out_dir / "TRACK_REPORT.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Circleworld seeded determinism and branch-active case generation.")
    parser.add_argument("--controller-manifest", default=str(DEFAULT_CONTROLLER_MANIFEST))
    parser.add_argument("--branch-first", default=str(DEFAULT_BRANCH_FIRST))
    parser.add_argument("--diversity-reference", default=str(DEFAULT_DIVERSITY_REFERENCE))
    parser.add_argument("--signature-init", default=str(DEFAULT_SIGNATURE_INIT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--phase-repeats", type=int, default=3)
    parser.add_argument("--rafa-init-seed", type=int, default=DEFAULT_SEED_RAFA_INIT)
    args = parser.parse_args()

    result = run_audit(
        controller_manifest=Path(args.controller_manifest),
        branch_first=Path(args.branch_first),
        diversity_reference=Path(args.diversity_reference),
        signature_init=Path(args.signature_init),
        out_dir=Path(args.out_dir),
        device_name=args.device,
        repeats=max(1, int(args.repeats)),
        phase_repeats=max(1, int(args.phase_repeats)),
        rafa_init_seed=int(args.rafa_init_seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
