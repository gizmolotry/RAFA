
import os
import subprocess
import re
import argparse
import time
import json
import sys
from datetime import datetime
from pathlib import Path

def run_grid(name, ckpt_path):
    seeds = [17, 23, 29, 41, 57, 83]
    steps_list = [8, 16, 32, 64]

    results = []
    repo_root = Path(__file__).resolve().parents[1]
    eval_dir = repo_root / "eval" / name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Regex to parse sample_diffusion output (robust to scientific notation)
    pattern = re.compile(
        r"rms=([\d\.eE-]+).*?top_ratio=([\d\.eE-]+).*?cent_var=([\d\.eE-]+).*?dphi_std=([\d\.eE-]+)"
        r"(?:.*?audio_effect=([\d\.eE-]+).*?state_effect=([\d\.eE-]+).*?coupling_score=([\d\.eE-]+))?"
    )

    for seed in seeds:
        for steps in steps_list:
            out_path = eval_dir / f"seed{seed}_s{steps}.wav"
            cmd = [
                sys.executable,
                "sample_diffusion.py",
                "--ckpt",
                str(ckpt_path),
                "--out",
                str(out_path),
                "--seed",
                str(seed),
                "--steps",
                str(steps),
                "--allow_collapse",
            ]

            match = None
            for attempt in range(3):
                print(f"Running (Attempt {attempt+1}): {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
                output = result.stdout + result.stderr

                match = pattern.search(output)
                if match:
                    break
                else:
                    print(f"Attempt {attempt+1} failed to parse output.")
                    if attempt < 2:
                        time.sleep(2)
            
            if match:
                rms = float(match.group(1))
                top_ratio = float(match.group(2))
                cent_var = float(match.group(3))
                dphi_std = float(match.group(4))
                audio_effect = float(match.group(5)) if match.group(5) is not None else 0.0
                state_effect = float(match.group(6)) if match.group(6) is not None else 0.0
                coupling_score = float(match.group(7)) if match.group(7) is not None else 0.0
                
                collapse = rms < 0.001 or top_ratio > 50.0 or cent_var < 100.0
                
                results.append({
                    "seed": seed,
                    "steps": steps,
                    "rms": rms,
                    "top_ratio": top_ratio,
                    "cent_var": cent_var,
                    "dphi_std": dphi_std,
                    "audio_effect": audio_effect,
                    "state_effect": state_effect,
                    "coupling_score": coupling_score,
                    "collapse": collapse
                })
            else:
                print(f"Failed to parse output for seed={seed} steps={steps}")
                if "ValueError" in output:
                    print("Error detected in sample_diffusion output.")
    
    return results

def log_to_ledger(name, ckpt_path, results, notes=""):
    if not results:
        print("No results to log.")
        return None
        
    total = len(results)
    collapses = sum(1 for r in results if r['collapse'])
    col_rate = (collapses / total) * 100
    
    avg_top = sum(r['top_ratio'] for r in results) / total
    avg_cent = sum(r['cent_var'] for r in results) / total
    avg_dphi = sum(r['dphi_std'] for r in results) / total
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"
        
    try:
        # Extract step count from checkpoint name
        step_match = re.search(r"step(\d+)", ckpt_path)
        if step_match:
            ckpt_steps = step_match.group(1)
        else:
            ep_match = re.search(r"ep(\d+)", ckpt_path)
            ckpt_steps = f"ep{ep_match.group(1)}" if ep_match else "unknown"
    except Exception:
        ckpt_steps = "unknown"

    row = f"| {date_str} | `{git_sha}` | {name} | {ckpt_steps} | {col_rate:.1f}% | {avg_top:.2f} | {avg_cent:.0f} | {avg_dphi:.4f} | {notes} |\n"

    with open("EVAL_LEDGER.md", "a", encoding="utf-8") as f:
        f.write(row)
    
    print(f"\nAppended to EVAL_LEDGER.md:")
    print(row.strip())
    return {
        "date": date_str,
        "git_sha": git_sha,
        "name": name,
        "ckpt_steps": ckpt_steps,
        "collapse_rate_pct": col_rate,
        "avg_top_ratio": avg_top,
        "avg_cent_var": avg_cent,
        "avg_dphi_std": avg_dphi,
        "notes": notes,
        "row": row.strip(),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Run or ablation name (e.g., baseline, rafa_full)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--notes", type=str, default="", help="Configuration notes or ablation details")
    args = parser.parse_args()

    results = run_grid(args.name, args.ckpt)
    
    print(f"\n--- {args.name.upper()} SCORECARD ---")
    header = (
        f"{'Seed':<6} | {'Steps':<6} | {'RMS':<8} | {'TopRat':<8} | {'CentVar':<8} | "
        f"{'DPhiStd':<8} | {'AudEff':<8} | {'StEff':<8} | {'Couple':<8} | {'Collapse'}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['seed']:<6} | {r['steps']:<6} | {r['rms']:<8.4f} | {r['top_ratio']:<8.2f} | "
            f"{r['cent_var']:<8.0f} | {r['dphi_std']:<8.4f} | {r['audio_effect']:<8.4f} | "
            f"{r['state_effect']:<8.4f} | {r['coupling_score']:<8.4f} | {int(r['collapse'])}"
        )

    ledger_entry = log_to_ledger(args.name, args.ckpt, results, args.notes)
    scorecard = {
        "name": args.name,
        "checkpoint": args.ckpt,
        "notes": args.notes,
        "num_results": len(results),
        "collapse_rate_pct": float(sum(1 for r in results if r["collapse"]) / max(1, len(results)) * 100.0) if results else None,
        "avg_top_ratio": float(sum(r["top_ratio"] for r in results) / max(1, len(results))) if results else None,
        "avg_cent_var": float(sum(r["cent_var"] for r in results) / max(1, len(results))) if results else None,
        "avg_dphi_std": float(sum(r["dphi_std"] for r in results) / max(1, len(results))) if results else None,
        "avg_audio_effect": float(sum(r["audio_effect"] for r in results) / max(1, len(results))) if results else None,
        "avg_state_effect": float(sum(r["state_effect"] for r in results) / max(1, len(results))) if results else None,
        "avg_coupling_score": float(sum(r["coupling_score"] for r in results) / max(1, len(results))) if results else None,
        "ledger_entry": ledger_entry,
        "results": results,
    }
    scorecard_path = Path("eval") / args.name / "scorecard.json"
    with scorecard_path.open("w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"Saved scorecard -> {scorecard_path}")
