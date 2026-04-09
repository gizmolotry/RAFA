import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
for path in (ROOT, CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib_blackwell import NakedRAFA
from rafa_math_tools import phase_to_phasor

from circleworld import (
    CircleworldConfig,
    circleworld_loss,
    recurse_circleworld,
    summarize_circleworld_run,
)


def _load_cfg() -> dict:
    cfg_path = Path(__file__).with_name("replacement_config.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_circleworld_cfg(cfg: dict, args: argparse.Namespace) -> CircleworldConfig:
    raw = cfg.get("circleworld", {})
    return CircleworldConfig(
        qset=tuple(raw.get("qset", (2, 3, 4, 5, 6, 8, 12))),
        q_weights=tuple(raw.get("q_weights", (1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5))),
        promotion_threshold=float(raw.get("promotion_threshold", 0.58)),
        max_promotions=int(raw.get("max_promotions", 4)),
        recursion_depth=int(args.depth if args.depth is not None else raw.get("recursion_depth", 2)),
        residue_scale=float(raw.get("residue_scale", 0.75)),
        child_law_gain=float(raw.get("child_law_gain", 0.25)),
        attack_window=int(raw.get("attack_window", 8)),
        persistence_momentum=float(raw.get("persistence_momentum", 0.6)),
    )


def _generate_seed_phase(
    rafa_core: NakedRAFA | None,
    batch_size: int,
    time_steps: int,
    device: torch.device,
    seed_source: str,
) -> torch.Tensor:
    if seed_source == "synthetic":
        freq_bins = 129
        t = torch.linspace(0.0, 1.0, steps=time_steps, device=device).view(1, 1, -1)
        freqs = torch.linspace(1.0, 3.5, steps=freq_bins, device=device).view(1, -1, 1)
        phase = 2.0 * torch.pi * freqs * t
        phase = phase + 0.12 * torch.sin(2.0 * torch.pi * (freqs / freqs.max()) * 3.0 * t)
        phase = phase + 0.05 * torch.randn((batch_size, freq_bins, time_steps), device=device)
        return phase_to_phasor(phase)

    if rafa_core is None:
        raise ValueError("rafa_core is required when seed_source='naked_rafa'")

    freq_bins = int(rafa_core.freq_bins)
    dummy_mag = torch.full((batch_size, freq_bins, time_steps), 0.1, device=device)
    dummy_phase = torch.randn((batch_size, freq_bins, time_steps), device=device)
    _, _, ext = rafa_core.forward(dummy_mag, dummy_phase, num_steps=6)
    return ext["phase_state"]


def run_formalization_ablation(
    mode: str,
    depth: int | None = None,
    batch_size: int = 2,
    time_steps: int = 96,
    device: str = "cuda",
    seed_source: str = "auto",
) -> dict:
    cfg = _load_cfg()
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    circle_cfg = _build_circleworld_cfg(cfg, argparse.Namespace(depth=depth))
    resolved_seed_source = seed_source
    if resolved_seed_source == "auto":
        resolved_seed_source = "naked_rafa" if dev.type == "cuda" else "synthetic"

    rafa_core = NakedRAFA(dev=dev.type) if resolved_seed_source == "naked_rafa" else None

    phase_state = _generate_seed_phase(
        rafa_core=rafa_core,
        batch_size=batch_size,
        time_steps=time_steps,
        device=dev,
        seed_source=resolved_seed_source,
    )
    run = recurse_circleworld(phase_state, cfg=circle_cfg, depth=depth, mode=mode)
    losses = circleworld_loss(run)
    summary = summarize_circleworld_run(run)

    packet_counts = [len(level) for level in run["packets"]]
    result = {
        "mode": mode,
        "depth": int(circle_cfg.recursion_depth if depth is None else depth),
        "batch_size": int(batch_size),
        "time_steps": int(time_steps),
        "device": str(dev),
        "seed_source": resolved_seed_source,
        **summary,
        "packet_counts_by_depth": packet_counts,
        "loss": float(losses["loss"].item()),
        "loss_terms": {k: float(v.item()) for k, v in losses.items() if k != "loss"},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RAFA Circleworld / Positive Replacement ablation.")
    parser.add_argument("--mode", default="active_packets", choices=["no_promotion", "passive_packets", "active_packets"])
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--time-steps", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-source", default="auto", choices=["auto", "naked_rafa", "synthetic"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = run_formalization_ablation(
        mode=args.mode,
        depth=args.depth,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
        device=args.device,
        seed_source=args.seed_source,
    )

    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
