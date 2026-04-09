
import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config
from phase_native_ifs import PhaseNativeIFS

def get_peak_energies(z, qset):
    """
    Computes the Ramanujan coherence for each q in qset.
    z: [B, Q, 2]
    """
    B, Q, _ = z.shape
    # Compute relative phasors r_ij
    zi = z.unsqueeze(2) # [B, Q, 1, 2]
    zj = z.unsqueeze(1) # [B, 1, Q, 2]
    # conj(zj)
    zj_conj = torch.stack([zj[..., 0], -zj[..., 1]], dim=-1)
    
    # r = zi * zj_conj
    r_re = zi[..., 0] * zj_conj[..., 0] - zi[..., 1] * zj_conj[..., 1]
    r_im = zi[..., 0] * zj_conj[..., 1] + zi[..., 1] * zj_conj[..., 0]
    
    # angle theta
    theta = torch.atan2(r_im, r_re)
    
    energies = {}
    for q in qset:
        # Coherence = Mean(cos(q * theta))
        coh = torch.cos(float(q) * theta).mean(dim=(1, 2))
        energies[int(q)] = coh # [B]
    return energies

def test_logic(args):
    # Deterministic seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup RAFA gru
    default_target_q = math.lcm(int(args.q1), int(args.q2))
    target_q = int(args.target_q) if args.target_q is not None else default_target_q
    q_logic_test = sorted(set([2, 3, 4, 5, 6, 8, 9, 10, 12, int(args.q1), int(args.q2), int(target_q)]))
    gru_ckpt = None
    if args.ckpt:
        print(f"Loading weights from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        gru_ckpt = ckpt
        if "config" in ckpt:
            cfg = ckpt["config"]
    cfg.setdefault("phase_native_ifs", {})
    if args.h0_anchor_enabled:
        cfg["phase_native_ifs"]["h0_anchor_enabled"] = True
    if args.no_h0_anchor_enabled:
        cfg["phase_native_ifs"]["h0_anchor_enabled"] = False
    if args.num_steps is not None:
        cfg["phase_native_ifs"]["num_steps"] = int(args.num_steps)
    if args.h0_anchor_lambda is not None:
        cfg["phase_native_ifs"]["h0_anchor_lambda"] = float(args.h0_anchor_lambda)
    if args.intermediate_consistency_enabled:
        cfg["phase_native_ifs"]["intermediate_consistency_enabled"] = True
    if args.no_intermediate_consistency_enabled:
        cfg["phase_native_ifs"]["intermediate_consistency_enabled"] = False
    if args.h0_anchor_decay is not None:
        cfg["phase_native_ifs"]["h0_anchor_decay"] = float(args.h0_anchor_decay)
    if args.h0_anchor_lambda_min is not None:
        cfg["phase_native_ifs"]["h0_anchor_lambda_min"] = float(args.h0_anchor_lambda_min)
    if args.intermediate_consistency_detach_target:
        cfg["phase_native_ifs"]["intermediate_consistency_detach_target"] = True
    if args.no_intermediate_consistency_detach_target:
        cfg["phase_native_ifs"]["intermediate_consistency_detach_target"] = False

    if args.ckpt:
        freq_bins = int(cfg["data"]["stft"]["n_fft"]) // 2 + 1
        gru = PhaseNativeIFS(q_bins=freq_bins, cfg=cfg.get("phase_native_ifs", {})).to(dev)
        # Handle different checkpoint formats
        state = ckpt["model"] if "model" in ckpt else ckpt
        # Filter for gru weights
        gru_state = {
            k.replace("rafa.phase_gru.", ""): v
            for k, v in state.items()
            if "phase_gru" in k and not k.endswith("qset_t") and not k.endswith("qw_t")
        }
        gru.load_state_dict(gru_state, strict=False)
    else:
        freq_bins = int(cfg["data"]["stft"]["n_fft"]) // 2 + 1
        gru = PhaseNativeIFS(q_bins=freq_bins, cfg=cfg.get("phase_native_ifs", {})).to(dev)
    gru.eval()

    # Initialize State: Inject two premises (e.g. q=2 and q=3)
    B = 1
    Q = gru.q_bins
    z = torch.randn(B, Q, 2, device=dev)
    # Normalize to unit phasors
    z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-12)
    
    # Strong injection of premises
    t = torch.linspace(0, 1, Q, device=dev)
    # Premise 1: q=2
    z_p1 = torch.stack([torch.cos(2*np.pi*args.q1*t), torch.sin(2*np.pi*args.q1*t)], dim=-1)
    # Premise 2: q=3
    z_p2 = torch.stack([torch.cos(2*np.pi*args.q2*t), torch.sin(2*np.pi*args.q2*t)], dim=-1)
    
    # z: [Q, 2] -> [1, Q, 2]
    z = z[0] + args.inject_strength * (z_p1 + z_p2)
    z = z.unsqueeze(0) # [1, Q, 2]
    z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-12)

    history = []
    ic_regs = []
    anchor_lambda_means = []
    
    print(f"Logic Task: {args.q1} AND {args.q2} -> {target_q}?")
    
    for i in range(args.iterations):
        with torch.no_grad():
            # Step the IFS gru
            z, debug = gru(z)
        if isinstance(debug, dict):
            if torch.is_tensor(debug.get("intermediate_consistency_reg", None)):
                ic_regs.append(float(debug["intermediate_consistency_reg"].item()))
            if "anchor_lambda_mean" in debug:
                anchor_lambda_means.append(float(debug["anchor_lambda_mean"]))
            
        # Analysis
        energies = get_peak_energies(z, q_logic_test)
        step_stats = {q: float(v.item()) for q, v in energies.items()}
        
        # Sort by energy to find rank
        sorted_qs = sorted(step_stats.items(), key=lambda x: x[1], reverse=True)
        # 1-based rank: 1 means strongest peak.
        rank = next((idx + 1 for idx, (q, e) in enumerate(sorted_qs) if q == target_q), 999)
        
        # Margin over next best non-target
        top_non_target = 0.0
        for q, e in sorted_qs:
            if q != target_q:
                top_non_target = float(e)
                break
        margin = (step_stats[target_q] / (top_non_target + 1e-12)) - 1.0
        
        history.append({
            "step": i,
            "energies": step_stats,
            "rank": rank,
            "margin": margin
        })

    # WIN CONDITION EVALUATION (PHD LEVEL)
    # 1. Rank: target in top-k (k=3)
    # 2. Margin: >= 0.15
    # 3. Persistence: last 10 steps
    
    last_n = min(15, args.iterations)
    recent = history[-last_n:]
    
    pers_count = sum(1 for h in recent if h["rank"] <= 3 and h["margin"] >= 0.15)
    persistent = pers_count >= (last_n * 0.6) # 60% of last steps
    
    final_rank = history[-1]['rank']
    final_margin = history[-1]['margin']
    final_energy = history[-1]['energies'][target_q]
    init_energy = history[0]['energies'][target_q]
    gain = final_energy > init_energy
    
    success = persistent and gain and final_rank <= 3

    model_tag = str(args.model_tag).strip() if args.model_tag else ""
    if not model_tag:
        if args.ckpt:
            model_tag = Path(args.ckpt).parent.name or Path(args.ckpt).stem
        else:
            model_tag = "random_init"
    
    result = {
        "success": success,
        "q1": args.q1,
        "q2": args.q2,
        "target_q": target_q,
        "model_tag": model_tag,
        "final_rank": final_rank,
        "final_margin": final_margin,
        "persistence_score": pers_count / last_n,
        "energy_gain": final_energy - init_energy,
        "mean_intermediate_consistency_reg": float(np.mean(ic_regs)) if ic_regs else 0.0,
        "mean_anchor_lambda": float(np.mean(anchor_lambda_means)) if anchor_lambda_means else 0.0,
        "timestamp": datetime.now().isoformat(),
        "q_bins": int(Q),
        "checkpoint_loaded": bool(gru_ckpt is not None),
    }
    
    print("\n--- LOGIC RESULT ---")
    print(json.dumps(result, indent=2))
    
    # Save to artifacts
    os.makedirs("research_track/logic_results", exist_ok=True)
    run_id = f"logic_{model_tag}_{args.q1}x{args.q2}_t{target_q}_s{args.seed}"
    with open(f"research_track/logic_results/{run_id}.json", "w") as f:
        json.dump({"args": vars(args), "result": result, "history": history}, f, indent=2)
        
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1", type=int, default=2)
    parser.add_argument("--q2", type=int, default=3)
    parser.add_argument("--target_q", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--inject_strength", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--h0_anchor_enabled", action="store_true")
    parser.add_argument("--no_h0_anchor_enabled", action="store_true")
    parser.add_argument("--h0_anchor_lambda", type=float, default=None)
    parser.add_argument("--h0_anchor_lambda_min", type=float, default=None)
    parser.add_argument("--h0_anchor_decay", type=float, default=None)
    parser.add_argument("--intermediate_consistency_enabled", action="store_true")
    parser.add_argument("--no_intermediate_consistency_enabled", action="store_true")
    parser.add_argument("--intermediate_consistency_detach_target", action="store_true")
    parser.add_argument("--no_intermediate_consistency_detach_target", action="store_true")
    args = parser.parse_args()
    test_logic(args)
