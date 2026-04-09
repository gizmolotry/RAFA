from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config
from phase_native_ifs import PhaseNativeIFS


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        if "x" in p:
            a, b = p.split("x", 1)
        elif ":" in p:
            a, b = p.split(":", 1)
        else:
            raise ValueError(f"Bad pair token: {p}")
        out.append((int(a), int(b)))
    if not out:
        raise ValueError("No pairs parsed.")
    return out


def _parse_qs(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("No q values parsed.")
    return sorted(set(vals))


def _gru_state_from_model_state(model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in model_state.items():
        if "phase_gru" not in k:
            continue
        kk = k.replace("rafa.phase_gru.", "")
        if kk.endswith("qset_t") or kk.endswith("qw_t"):
            continue
        out[kk] = v
    return out


def load_gru(
    *,
    device: torch.device,
    ckpt_path: str | None,
    base_cfg_path: str | None,
    random_init: bool,
) -> tuple[PhaseNativeIFS, dict[str, Any], str]:
    if ckpt_path is not None:
        ck = torch.load(ckpt_path, map_location="cpu")
        cfg = ck.get("config", load_config(base_cfg_path) if base_cfg_path else load_config())
        source = ckpt_path
        freq_bins = int(cfg["data"]["stft"]["n_fft"]) // 2 + 1
        gru = PhaseNativeIFS(freq_bins, cfg.get("phase_native_ifs", {})).to(device)
        if not random_init:
            state = ck.get("model", ck)
            gru_sd = _gru_state_from_model_state(state)
            if gru_sd:
                gru.load_state_dict(gru_sd, strict=False)
        return gru, cfg, source

    cfg = load_config(base_cfg_path) if base_cfg_path else load_config()
    freq_bins = int(cfg["data"]["stft"]["n_fft"]) // 2 + 1
    gru = PhaseNativeIFS(freq_bins, cfg.get("phase_native_ifs", {})).to(device)
    return gru, cfg, "config_random_init"


def _normalize(z: torch.Tensor) -> torch.Tensor:
    return z / torch.sqrt((z * z).sum(dim=-1, keepdim=True).clamp_min(1e-12))


def build_injection(
    *,
    q1: int,
    q2: int,
    q_bins: int,
    inject_strength: float,
    device: torch.device,
) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, q_bins, device=device).unsqueeze(0)
    phase1 = 2.0 * math.pi * float(q1) * t + (2.0 * math.pi * torch.rand(1, 1, device=device))
    phase2 = 2.0 * math.pi * float(q2) * t + (2.0 * math.pi * torch.rand(1, 1, device=device))
    z1 = torch.stack([torch.cos(phase1), torch.sin(phase1)], dim=-1)
    z2 = torch.stack([torch.cos(phase2), torch.sin(phase2)], dim=-1)
    noise = _normalize(torch.randn(1, q_bins, 2, device=device))
    z = noise + float(inject_strength) * (z1 + z2)
    return _normalize(z)


def build_batch(
    *,
    pairs: list[tuple[int, int]],
    batch_size: int,
    q_bins: int,
    inject_strength: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], list[int], list[int]]:
    zs = []
    q1s: list[int] = []
    q2s: list[int] = []
    targets: list[int] = []
    for _ in range(batch_size):
        q1, q2 = random.choice(pairs)
        zs.append(build_injection(q1=q1, q2=q2, q_bins=q_bins, inject_strength=inject_strength, device=device))
        q1s.append(int(q1))
        q2s.append(int(q2))
        targets.append(int(math.lcm(int(q1), int(q2))))
    z = torch.cat(zs, dim=0)
    return z, q1s, q2s, targets


def rollout(gru: PhaseNativeIFS, z: torch.Tensor, steps: int) -> torch.Tensor:
    x = z
    for _ in range(int(steps)):
        x, _ = gru(x)
    return x


def q_energies(z: torch.Tensor, q_values: list[int]) -> torch.Tensor:
    zi = z.unsqueeze(2)
    zj = z.unsqueeze(1)
    zj_conj = torch.stack([zj[..., 0], -zj[..., 1]], dim=-1)
    r_re = zi[..., 0] * zj_conj[..., 0] - zi[..., 1] * zj_conj[..., 1]
    r_im = zi[..., 0] * zj_conj[..., 1] + zi[..., 1] * zj_conj[..., 0]
    theta = torch.atan2(r_im, r_re)
    vals = []
    for q in q_values:
        vals.append(torch.cos(float(q) * theta).mean(dim=(1, 2)))
    return torch.stack(vals, dim=1)


def set_train_scope(gru: PhaseNativeIFS, scope: str) -> None:
    for p in gru.parameters():
        p.requires_grad = False

    if scope == "all":
        for p in gru.parameters():
            p.requires_grad = True
        return
    if scope == "router_only":
        for p in gru.router.parameters():
            p.requires_grad = True
        return
    if scope == "maps_only":
        for m in gru.maps:
            for p in m.parameters():
                p.requires_grad = True
        return
    if scope == "router_maps":
        for p in gru.router.parameters():
            p.requires_grad = True
        for m in gru.maps:
            for p in m.parameters():
                p.requires_grad = True
        return
    if scope == "slow_only":
        for p in gru.slow_gru.parameters():
            p.requires_grad = True
        for p in gru.slow_proj.parameters():
            p.requires_grad = True
        if hasattr(gru, "memory"):
            for p in gru.memory.parameters():
                p.requires_grad = True
        return
    raise ValueError(f"Unknown train scope: {scope}")


def train_probe(
    *,
    gru: PhaseNativeIFS,
    pairs: list[tuple[int, int]],
    q_values: list[int],
    steps: int,
    batch_size: int,
    rollout_steps: int,
    inject_strength: float,
    lr: float,
    margin: float,
    margin_weight: float,
    decoy_q: int,
    decoy_weight: float,
    scope: str,
    device: torch.device,
) -> dict[str, Any]:
    gru.train()
    set_train_scope(gru, scope)
    params = [p for p in gru.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters in selected scope.")
    opt = torch.optim.Adam(params, lr=float(lr))
    q_to_idx = {q: i for i, q in enumerate(q_values)}
    if decoy_q not in q_to_idx:
        decoy_q = q_values[0]
    decoy_idx = q_to_idx[decoy_q]

    logs = []
    for step in range(1, int(steps) + 1):
        z0, _, _, targets = build_batch(
            pairs=pairs,
            batch_size=batch_size,
            q_bins=gru.q_bins,
            inject_strength=inject_strength,
            device=device,
        )
        zf = rollout(gru, z0, rollout_steps)
        energies = q_energies(zf, q_values=q_values)  # [B,M]

        t_idx = torch.tensor([q_to_idx[t] for t in targets], device=device, dtype=torch.long)
        ce = F.cross_entropy(energies, t_idx)
        t_val = energies.gather(1, t_idx.unsqueeze(1)).squeeze(1)
        non_target = energies.clone()
        non_target[torch.arange(non_target.size(0), device=device), t_idx] = -1e9
        top_non_target = non_target.max(dim=1).values
        mloss = F.relu(top_non_target - t_val + float(margin)).mean()

        decoy = energies[:, decoy_idx]
        decoy_mask = (t_idx != decoy_idx).float()
        dloss = (decoy * decoy_mask).mean()

        loss = ce + float(margin_weight) * mloss + float(decoy_weight) * dloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(1, steps // 10) == 0 or step == 1 or step == steps:
            with torch.no_grad():
                pred = energies.argmax(dim=1)
                acc = (pred == t_idx).float().mean().item()
                logs.append(
                    {
                        "step": step,
                        "loss": float(loss.item()),
                        "ce": float(ce.item()),
                        "margin_loss": float(mloss.item()),
                        "decoy_loss": float(dloss.item()),
                        "batch_acc": float(acc),
                    }
                )

    return {"train_logs": logs, "steps": int(steps)}


def evaluate_single(
    *,
    gru: PhaseNativeIFS,
    q_values: list[int],
    q1: int,
    q2: int,
    target_q: int,
    seed: int,
    rollout_steps: int,
    inject_strength: float,
    device: torch.device,
) -> dict[str, Any]:
    _set_seed(seed)
    gru.eval()
    z = build_injection(q1=q1, q2=q2, q_bins=gru.q_bins, inject_strength=inject_strength, device=device)

    history = []
    drift_values: list[float] = []
    ic_values: list[float] = []
    prev_energy_vec: torch.Tensor | None = None
    with torch.no_grad():
        for step in range(int(rollout_steps)):
            z, debug = gru(z)
            e = q_energies(z, q_values=q_values)[0]
            if prev_energy_vec is not None:
                drift_values.append(float((e - prev_energy_vec).abs().mean().item()))
            prev_energy_vec = e.detach().clone()
            if isinstance(debug, dict) and torch.is_tensor(debug.get("intermediate_consistency_reg", None)):
                ic_values.append(float(debug["intermediate_consistency_reg"].item()))
            stat = {q: float(ei.item()) for q, ei in zip(q_values, e)}
            ranked = sorted(stat.items(), key=lambda kv: kv[1], reverse=True)
            rank = next((i + 1 for i, (q, _) in enumerate(ranked) if q == int(target_q)), 999)
            top_non = next((v for q, v in ranked if q != int(target_q)), 0.0)
            margin = (stat[int(target_q)] / (top_non + 1e-12)) - 1.0
            history.append({"step": step, "rank": rank, "margin": margin, "energies": stat})

    last_n = min(15, len(history))
    recent = history[-last_n:]
    pers_count = sum(1 for h in recent if h["rank"] <= 3 and h["margin"] >= 0.15)
    persistent = pers_count >= (0.6 * last_n)
    final = history[-1]
    init_e = history[0]["energies"][int(target_q)]
    fin_e = final["energies"][int(target_q)]
    gain = fin_e > init_e
    success = bool(persistent and gain and final["rank"] <= 3)
    top_q = int(max(final["energies"].items(), key=lambda kv: kv[1])[0])
    return {
        "success": success,
        "final_rank": int(final["rank"]),
        "final_margin": float(final["margin"]),
        "persistence_score": float(pers_count / max(1, last_n)),
        "energy_gain": float(fin_e - init_e),
        "top_q": top_q,
        "target_energy_final": float(fin_e),
        "mean_q_drift": float(np.mean(drift_values)) if drift_values else 0.0,
        "mean_intermediate_consistency_reg": float(np.mean(ic_values)) if ic_values else 0.0,
    }


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in results:
        if "error" in r:
            continue
        key = (str(r["control"]), str(r.get("case", "default")), str(r["model"]))
        grouped.setdefault(key, []).append(r)

    out = []
    for (control, case, model), rows in sorted(grouped.items()):
        n = len(rows)
        out.append(
            {
                "control": control,
                "case": case,
                "model": model,
                "n": n,
                "win_rate": float(sum(int(x["success"]) for x in rows) / max(1, n)),
                "mean_final_rank": float(np.mean([x["final_rank"] for x in rows])) if rows else 999.0,
                "mean_final_margin": float(np.mean([x["final_margin"] for x in rows])) if rows else 0.0,
                "mean_persistence": float(np.mean([x["persistence_score"] for x in rows])) if rows else 0.0,
                "mean_energy_gain": float(np.mean([x["energy_gain"] for x in rows])) if rows else 0.0,
                "q5_top_rate": float(np.mean([1.0 if x["top_q"] == 5 else 0.0 for x in rows])) if rows else 0.0,
                "mean_q_drift": float(np.mean([x.get("mean_q_drift", 0.0) for x in rows])) if rows else 0.0,
                "mean_intermediate_consistency_reg": float(np.mean([x.get("mean_intermediate_consistency_reg", 0.0) for x in rows])) if rows else 0.0,
            }
        )
    return out


def run_controls(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    _set_seed(args.seed)

    train_pairs = _parse_pairs(args.train_pairs)
    q_values = _parse_qs(args.q_values)
    required_targets = {math.lcm(int(a), int(b)) for a, b in train_pairs}
    required_targets.update({6, 4, 12})  # Control A/C targets.
    q_values = sorted(set(q_values).union(required_targets))
    ablation_specs = {
        "rafa_full": args.ckpt_full,
        "rafa_no_memory": args.ckpt_no_memory,
        "rafa_no_ramanujan": args.ckpt_no_ramanujan,
        "rafa_no_slow_clock": args.ckpt_no_slow_clock,
    }

    trained_grus: dict[str, PhaseNativeIFS] = {}

    # Train per source (manifold-native supervised probe)
    for name, ckpt in ablation_specs.items():
        gru, _, _ = load_gru(device=device, ckpt_path=ckpt, base_cfg_path=args.base_config, random_init=False)
        train_probe(
            gru=gru,
            pairs=train_pairs,
            q_values=q_values,
            steps=args.probe_steps,
            batch_size=args.batch_size,
            rollout_steps=args.train_rollout_steps,
            inject_strength=args.inject_strength,
            lr=args.lr,
            margin=args.margin,
            margin_weight=args.margin_weight,
            decoy_q=args.decoy_q,
            decoy_weight=args.decoy_weight,
            scope=args.train_scope,
            device=device,
        )
        trained_grus[name] = gru

    random_gru, _, _ = load_gru(
        device=device,
        ckpt_path=args.ckpt_full,
        base_cfg_path=args.base_config,
        random_init=True,
    )
    train_probe(
        gru=random_gru,
        pairs=train_pairs,
        q_values=q_values,
        steps=args.probe_steps,
        batch_size=args.batch_size,
        rollout_steps=args.train_rollout_steps,
        inject_strength=args.inject_strength,
        lr=args.lr,
        margin=args.margin,
        margin_weight=args.margin_weight,
        decoy_q=args.decoy_q,
        decoy_weight=args.decoy_weight,
        scope=args.train_scope,
        device=device,
    )
    trained_grus["random_init"] = random_gru

    results: list[dict[str, Any]] = []

    # Control A: trained full vs trained random on 2x3->6
    for seed in range(args.eval_seeds):
        for name in ["rafa_full", "random_init"]:
            m = evaluate_single(
                gru=trained_grus[name],
                q_values=q_values,
                q1=2,
                q2=3,
                target_q=6,
                seed=args.seed + 1000 + seed,
                rollout_steps=args.eval_rollout_steps,
                inject_strength=args.inject_strength,
                device=device,
            )
            results.append({"control": "A", "model": name, **m})

    # Control B: ablations on 2x3->6
    for seed in range(args.eval_seeds):
        for name in ["rafa_full", "rafa_no_memory", "rafa_no_ramanujan", "rafa_no_slow_clock"]:
            m = evaluate_single(
                gru=trained_grus[name],
                q_values=q_values,
                q1=2,
                q2=3,
                target_q=6,
                seed=args.seed + 2000 + seed,
                rollout_steps=args.eval_rollout_steps,
                inject_strength=args.inject_strength,
                device=device,
            )
            results.append({"control": "B", "model": name, **m})

    # Control C: negative/alternate targets
    neg_cases = [(2, 4, 4), (3, 6, 6), (4, 6, 12)]
    for seed in range(args.eval_seeds):
        for q1, q2, target in neg_cases:
            for name in ["rafa_full", "rafa_no_ramanujan", "random_init"]:
                m = evaluate_single(
                    gru=trained_grus[name],
                    q_values=q_values,
                    q1=q1,
                    q2=q2,
                    target_q=target,
                    seed=args.seed + 3000 + seed,
                    rollout_steps=args.eval_rollout_steps,
                    inject_strength=args.inject_strength,
                    device=device,
                )
                results.append({"control": "C", "case": f"{q1}_{q2}_to_{target}", "model": name, **m})

    summary = _summary_rows(results)
    out = {
        "settings": {
            "train_pairs": train_pairs,
            "q_values": q_values,
            "probe_steps": args.probe_steps,
            "eval_seeds": args.eval_seeds,
            "train_scope": args.train_scope,
            "lr": args.lr,
        },
        "num_results": len(results),
        "results": results,
        "summary": summary,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")
    for row in summary:
        print(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Manifold-native supervised probe for RAFA gru.")
    ap.add_argument("--ckpt_full", type=str, default="checkpoints_diffusion_rafa_full/diff_step1000.pt")
    ap.add_argument("--ckpt_no_memory", type=str, default="checkpoints_diffusion_rafa_no_memory/diff_step1000.pt")
    ap.add_argument("--ckpt_no_ramanujan", type=str, default="checkpoints_diffusion_rafa_no_ramanujan/diff_step1000.pt")
    ap.add_argument("--ckpt_no_slow_clock", type=str, default="checkpoints_diffusion_rafa_no_slow_clock/diff_step1000.pt")
    ap.add_argument("--base_config", type=str, default=None)
    ap.add_argument("--train_pairs", type=str, default="2x3,2x4,3x6,4x6,3x4,5x6")
    ap.add_argument("--q_values", type=str, default="2,3,4,5,6,8,9,10,12")
    ap.add_argument("--probe_steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--train_rollout_steps", type=int, default=12)
    ap.add_argument("--eval_rollout_steps", type=int, default=100)
    ap.add_argument("--eval_seeds", type=int, default=20)
    ap.add_argument("--inject_strength", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--margin_weight", type=float, default=1.0)
    ap.add_argument("--decoy_q", type=int, default=5)
    ap.add_argument("--decoy_weight", type=float, default=0.2)
    ap.add_argument("--train_scope", type=str, default="router_maps", choices=["all", "router_only", "maps_only", "router_maps", "slow_only"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", type=str, default="eval/relational_probe_controls_summary.json")
    args = ap.parse_args()
    run_controls(args)


if __name__ == "__main__":
    main()
