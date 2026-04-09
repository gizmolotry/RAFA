from __future__ import annotations

import argparse
import csv
import json
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

from ablate_formalization import _generate_seed_phase
from circleworld import CircleworldConfig, circleworld_loss, recurse_circleworld, summarize_circleworld_run
from lib_blackwell import NakedRAFA


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _heldout_plan(device: torch.device) -> list[tuple[str, int]]:
    plan: list[tuple[str, int]] = [("synthetic", 5100 + i) for i in range(6)]
    if device.type == "cuda":
        plan.extend([("naked_rafa", 6100 + i) for i in range(3)])
    return plan


def _trajectory_rows(run: dict[str, Any], source: str, seed: int) -> list[dict[str, Any]]:
    hist = run["history"]
    rows: list[dict[str, Any]] = []
    for depth_idx, block in enumerate(hist):
        major = block["major_mass"][0].detach().cpu()
        residue = block["minor_residue"][0].detach().cpu()
        promo = block["promotability"][0].detach().cpu()
        per_q = block["per_q"][0].abs().detach().cpu()  # [K, T]
        q_mass = per_q / per_q.sum(dim=0, keepdim=True).clamp_min(1e-8)
        dom = q_mass.max(dim=0).values
        entropy = -(q_mass * q_mass.clamp_min(1e-8).log()).sum(dim=0) / torch.log(torch.tensor(float(q_mass.size(0))))
        for t in range(major.numel()):
            rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "depth": depth_idx,
                    "t": t,
                    "major_mass": float(major[t].item()),
                    "minor_residue": float(residue[t].item()),
                    "promotability": float(promo[t].item()),
                    "dominant_q_share": float(dom[t].item()),
                    "q_entropy": float(entropy[t].item()),
                }
            )
    return rows


def evaluate_config(config_path: Path, out_dir: Path, time_steps: int, device_name: str) -> dict[str, Any]:
    device = _safe_device(device_name)
    cfg = _load_circle_cfg(config_path)
    plan = _heldout_plan(device)
    rafa_core: NakedRAFA | None = NakedRAFA(dev=device.type) if any(src == "naked_rafa" for src, _ in plan) else None

    rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    for source, seed in plan:
        phase_state = _generate_seed_phase(
            rafa_core=rafa_core,
            batch_size=1,
            time_steps=time_steps,
            device=device,
            seed_source=source,
        ).detach()
        run = recurse_circleworld(phase_state, cfg=cfg, depth=cfg.recursion_depth, mode="active_packets")
        summary = summarize_circleworld_run(run)
        losses = circleworld_loss(run)
        rows.append(
            {
                "source": source,
                "seed": seed,
                **summary,
                "loss": float(losses["loss"].item()),
                "l_q_dom": float(losses["l_q_dom"].item()),
                "l_q_entropy": float(losses["l_q_entropy"].item()),
                "l_major_sat": float(losses["l_major_sat"].item()),
            }
        )
        traj_rows.extend(_trajectory_rows(run, source=source, seed=seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "heldout_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        writer.writeheader()
        writer.writerows(traj_rows)

    summary = {
        "config_path": str(config_path),
        "device": str(device),
        "time_steps": time_steps,
        "num_samples": len(rows),
        "mean_major_gain": float(sum(r["major_gain"] for r in rows) / len(rows)),
        "mean_residue_drop": float(sum(r["residue_drop"] for r in rows) / len(rows)),
        "mean_promotability_gain": float(sum(r["promotability_gain"] for r in rows) / len(rows)),
        "mean_dominant_q_share": float(sum(r["dominant_q_share"] for r in rows) / len(rows)),
        "mean_q_entropy": float(sum(r["q_entropy"] for r in rows) / len(rows)),
        "mean_major_saturation": float(sum(r["major_saturation"] for r in rows) / len(rows)),
        "mean_loss": float(sum(r["loss"] for r in rows) / len(rows)),
        "mean_l_q_dom": float(sum(r["l_q_dom"] for r in rows) / len(rows)),
        "mean_l_q_entropy": float(sum(r["l_q_entropy"] for r in rows) / len(rows)),
        "mean_l_major_sat": float(sum(r["l_major_sat"] for r in rows) / len(rows)),
        "rows": rows,
        "trajectory_csv": str(csv_path),
    }
    (out_dir / "heldout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a Circleworld config on held-out seeds and export trajectories.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--time-steps", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    summary = evaluate_config(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        time_steps=args.time_steps,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
