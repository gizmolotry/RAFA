import argparse
import hashlib
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

DEFAULT_SEED_RAFA_INIT = 314159
DEFAULT_NAKED_RAFA_SEED_POLICY = "legacy"
DEFAULT_NAKED_RAFA_PHASE_ONLY_MAGNITUDE = 0.125
DEFAULT_NAKED_RAFA_MAG_SWEEP = (0.05, 0.125, 0.2)


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


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _phase_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    align = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).clamp(-1.0, 1.0)
    return (1.0 - align).clamp(0.0, 2.0)


def _record_magnitude_invariance_audit(
    audit_out: dict | None,
    phase_states: list[torch.Tensor],
    sweep: tuple[float, ...],
) -> None:
    if audit_out is None:
        return
    pairwise_mean = []
    pairwise_max = []
    for idx, left in enumerate(phase_states):
        for right in phase_states[idx + 1 :]:
            dist = _phase_distance(left, right)
            pairwise_mean.append(float(dist.mean().item()))
            pairwise_max.append(float(dist.amax().item()))
    audit_out["naked_rafa_magnitude_invariance"] = {
        "magnitude_levels": [float(v) for v in sweep],
        "phase_state_sha256_by_level": [_tensor_sha256(state) for state in phase_states],
        "mean_pairwise_phase_distance": float(sum(pairwise_mean) / max(1, len(pairwise_mean))),
        "max_pairwise_phase_distance": float(max(pairwise_max) if pairwise_max else 0.0),
        "num_pairwise_comparisons": int(len(pairwise_mean)),
    }


def make_seed_rafa(dev: str = "cuda", init_seed: int = DEFAULT_SEED_RAFA_INIT) -> NakedRAFA:
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(init_seed))
        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(init_seed))
        return NakedRAFA(dev=dev)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _generate_seed_phase(
    rafa_core: NakedRAFA | None,
    batch_size: int,
    time_steps: int,
    device: torch.device,
    seed_source: str,
    seed: int | None = None,
    naked_rafa_seed_policy: str = DEFAULT_NAKED_RAFA_SEED_POLICY,
    naked_rafa_phase_only_magnitude: float = DEFAULT_NAKED_RAFA_PHASE_ONLY_MAGNITUDE,
    naked_rafa_mag_sweep: tuple[float, ...] | None = None,
    audit_out: dict | None = None,
) -> torch.Tensor:
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

    def _seeded_randn(shape: tuple[int, ...]) -> torch.Tensor:
        if gen is None:
            return torch.randn(shape, device=device)
        return torch.randn(shape, generator=gen).to(device)

    def _seeded_rand(shape: tuple[int, ...]) -> torch.Tensor:
        if gen is None:
            return torch.rand(shape, device=device)
        return torch.rand(shape, generator=gen).to(device)

    if seed_source == "synthetic":
        freq_bins = 129
        t = torch.linspace(0.0, 1.0, steps=time_steps, device=device).view(1, 1, -1)
        freqs = torch.linspace(1.0, 3.5, steps=freq_bins, device=device).view(1, -1, 1)
        phase = 2.0 * torch.pi * freqs * t
        phase = phase + 0.12 * torch.sin(2.0 * torch.pi * (freqs / freqs.max()) * 3.0 * t)
        phase = phase + 0.05 * _seeded_randn((batch_size, freq_bins, time_steps))
        return phase_to_phasor(phase)

    if rafa_core is None:
        raise ValueError("rafa_core is required when seed_source='naked_rafa'")

    freq_bins = int(rafa_core.freq_bins)
    dummy_phase = _seeded_randn((batch_size, freq_bins, time_steps))
    if naked_rafa_seed_policy == "legacy":
        dummy_mag = 0.05 + 0.15 * _seeded_rand((batch_size, freq_bins, time_steps))
        _, _, ext = rafa_core.forward(dummy_mag, dummy_phase, num_steps=6)
        return ext["phase_state"]
    if naked_rafa_seed_policy == "phase_only":
        dummy_mag = torch.full(
            (batch_size, freq_bins, time_steps),
            float(naked_rafa_phase_only_magnitude),
            device=device,
        )
        _, _, ext = rafa_core.forward(dummy_mag, dummy_phase, num_steps=6)
        return ext["phase_state"]
    if naked_rafa_seed_policy == "mag_invariance_sweep":
        sweep = naked_rafa_mag_sweep or DEFAULT_NAKED_RAFA_MAG_SWEEP
        if not sweep:
            raise ValueError("naked_rafa_mag_sweep must contain at least one magnitude")
        phase_states = []
        for mag_level in sweep:
            dummy_mag = torch.full((batch_size, freq_bins, time_steps), float(mag_level), device=device)
            _, _, ext = rafa_core.forward(dummy_mag, dummy_phase, num_steps=6)
            phase_states.append(ext["phase_state"])
        _record_magnitude_invariance_audit(audit_out, phase_states, tuple(float(v) for v in sweep))
        phase_state = torch.stack(phase_states, dim=0).mean(dim=0)
        phase_state = phase_state / torch.linalg.vector_norm(phase_state, dim=-1, keepdim=True).clamp_min(1e-8)
        return phase_state
    raise ValueError(f"Unknown naked_rafa_seed_policy: {naked_rafa_seed_policy}")


def run_formalization_ablation(
    mode: str,
    depth: int | None = None,
    batch_size: int = 2,
    time_steps: int = 96,
    device: str = "cuda",
    seed_source: str = "auto",
    seed: int | None = None,
    rafa_init_seed: int = DEFAULT_SEED_RAFA_INIT,
    naked_rafa_seed_policy: str = DEFAULT_NAKED_RAFA_SEED_POLICY,
    naked_rafa_phase_only_magnitude: float = DEFAULT_NAKED_RAFA_PHASE_ONLY_MAGNITUDE,
    naked_rafa_mag_sweep: tuple[float, ...] | None = None,
) -> dict:
    cfg = _load_cfg()
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    circle_cfg = _build_circleworld_cfg(cfg, argparse.Namespace(depth=depth))
    resolved_seed_source = seed_source
    resolved_seed_policy = naked_rafa_seed_policy
    if resolved_seed_source == "auto":
        resolved_seed_source = "naked_rafa" if dev.type == "cuda" else "synthetic"
    elif resolved_seed_source == "naked_rafa_phase_only":
        resolved_seed_source = "naked_rafa"
        resolved_seed_policy = "phase_only"
    elif resolved_seed_source == "naked_rafa_mag_invariance_sweep":
        resolved_seed_source = "naked_rafa"
        resolved_seed_policy = "mag_invariance_sweep"

    rafa_core = make_seed_rafa(dev=dev.type, init_seed=rafa_init_seed) if resolved_seed_source == "naked_rafa" else None
    seed_audit: dict = {}

    phase_state = _generate_seed_phase(
        rafa_core=rafa_core,
        batch_size=batch_size,
        time_steps=time_steps,
        device=dev,
        seed_source=resolved_seed_source,
        seed=seed,
        naked_rafa_seed_policy=resolved_seed_policy,
        naked_rafa_phase_only_magnitude=naked_rafa_phase_only_magnitude,
        naked_rafa_mag_sweep=naked_rafa_mag_sweep,
        audit_out=seed_audit,
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
        "naked_rafa_seed_policy": resolved_seed_policy if resolved_seed_source == "naked_rafa" else None,
        "naked_rafa_phase_only_magnitude": (
            float(naked_rafa_phase_only_magnitude)
            if resolved_seed_source == "naked_rafa" and resolved_seed_policy == "phase_only"
            else None
        ),
        "naked_rafa_mag_sweep": (
            [float(v) for v in (naked_rafa_mag_sweep or DEFAULT_NAKED_RAFA_MAG_SWEEP)]
            if resolved_seed_source == "naked_rafa" and resolved_seed_policy == "mag_invariance_sweep"
            else None
        ),
        "seed": int(seed) if seed is not None else None,
        "rafa_init_seed": int(rafa_init_seed) if resolved_seed_source == "naked_rafa" else None,
        "phase_state_sha256": _tensor_sha256(phase_state),
        **seed_audit,
        **summary,
        "packet_counts_by_depth": packet_counts,
        "loss": float(losses["loss"].item()),
        "loss_terms": {k: float(v.item()) for k, v in losses.items() if k != "loss"},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RAFA Circleworld / Positive Replacement ablation.")
    parser.add_argument(
        "--mode",
        default="active_packets",
        choices=[
            "no_promotion",
            "passive_packets",
            "active_packets",
            "native_multimode",
            "native_multimode_childworld",
        ],
    )
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--time-steps", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed-source",
        default="auto",
        choices=["auto", "naked_rafa", "synthetic", "naked_rafa_phase_only", "naked_rafa_mag_invariance_sweep"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rafa-init-seed", type=int, default=DEFAULT_SEED_RAFA_INIT)
    parser.add_argument(
        "--naked-rafa-seed-policy",
        default=DEFAULT_NAKED_RAFA_SEED_POLICY,
        choices=["legacy", "phase_only", "mag_invariance_sweep"],
    )
    parser.add_argument("--naked-rafa-phase-only-magnitude", type=float, default=DEFAULT_NAKED_RAFA_PHASE_ONLY_MAGNITUDE)
    parser.add_argument(
        "--naked-rafa-mag-sweep",
        default=",".join(str(v) for v in DEFAULT_NAKED_RAFA_MAG_SWEEP),
        help="Comma-separated magnitude levels used when --naked-rafa-seed-policy=mag_invariance_sweep.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mag_sweep = tuple(float(v.strip()) for v in str(args.naked_rafa_mag_sweep).split(",") if v.strip())

    result = run_formalization_ablation(
        mode=args.mode,
        depth=args.depth,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
        device=args.device,
        seed_source=args.seed_source,
        seed=args.seed,
        rafa_init_seed=args.rafa_init_seed,
        naked_rafa_seed_policy=args.naked_rafa_seed_policy,
        naked_rafa_phase_only_magnitude=args.naked_rafa_phase_only_magnitude,
        naked_rafa_mag_sweep=mag_sweep,
    )

    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
