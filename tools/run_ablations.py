
import os
import subprocess
import yaml
import sys
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime

def get_manifest_hash(ablation_name, steps):
    with open("config.yaml", "r") as f:
        config_str = f.read()
    manifest_blob = f"{config_str}_{ablation_name}_{steps}"
    return hashlib.sha256(manifest_blob.encode()).hexdigest()

def update_config_for_ablation(ablation_name):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["phase_native_ifs"]["memory_enabled"] = (ablation_name != "rafa_no_memory")
    with open("config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def main():
    steps = 1000
    ablations = [
        {"name": "rafa_full", "flags": [], "notes": "Full RAFA-PC v3.0 (Memory + Hyena + Ramanujan + Slow Clock)"},
        {"name": "rafa_no_memory", "flags": [], "notes": "Ablation: Relational Motif Memory disabled"},
        {"name": "rafa_no_ramanujan", "flags": ["--ablate_ramanujan"], "notes": "Ablation: Ramanujan Gating disabled"},
        {"name": "rafa_no_slow_clock", "flags": ["--ablate_slow_clock"], "notes": "Ablation: Slow Clock (Multi-rate) disabled"}
    ]
    
    os.makedirs("logs/ablations", exist_ok=True)
    os.makedirs("research_track/artifacts", exist_ok=True)
    
    msvc_path = r"C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
    os.environ["PATH"] = msvc_path + os.pathsep + os.environ["PATH"]
    os.environ["PYTHONPATH"] = "."
    
    for ab in ablations:
        print(f"\n" + "="*43 + f"\nStarting Ablation: {ab['name']}\n" + "="*43)
        update_config_for_ablation(ab["name"])
        
        current_hash = get_manifest_hash(ab["name"], steps)
        ckpt_dir = f"checkpoints_diffusion_{ab['name']}"
        manifest_path = os.path.join(ckpt_dir, "run_manifest.json")
        
        should_train = True
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                old_manifest = json.load(f)
                if old_manifest.get("manifest_hash") == current_hash:
                    print(f"VALID MANIFEST: Resuming from existing checkpoint in {ckpt_dir}")
                    should_train = False
                else:
                    print(f"MANIFEST MISMATCH: Retraining {ab['name']}.")
        
        log_path = f"logs/ablations/train_{ab['name']}.log"
        os.makedirs(ckpt_dir, exist_ok=True)
        
        if should_train:
            train_cmd = [
                sys.executable, "-u", "train_diffusion.py",
                "--model_type", "rafa",
                "--max_steps", str(steps),
                "--ckpt_dir", ckpt_dir
            ] + ab["flags"]
            print(f"Executing: {' '.join(train_cmd)}")
            with open(log_path, "w", buffering=1) as f:
                subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT)
            
            with open(manifest_path, "w") as f:
                json.dump({
                    "name": ab["name"],
                    "steps": steps,
                    "manifest_hash": current_hash,
                    "timestamp": datetime.now().isoformat(),
                    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                }, f, indent=2)

        ckpt_path = os.path.join(ckpt_dir, f"diff_step{steps}.pt")
        if not os.path.exists(ckpt_path):
            files = sorted(list(Path(ckpt_dir).glob("*.pt")), key=os.path.getmtime)
            if files:
                ckpt_path = str(files[-1])
                print(f"Fallback checkpoint: {ckpt_path}")
            else:
                print(f"ERROR: Checkpoint missing for {ab['name']}.")
                continue
                
        # --- PATH B EVALUATION ---
        in_ledger = False
        if os.path.exists("EVAL_LEDGER.md"):
            with open("EVAL_LEDGER.md", "r") as f:
                content = f.read()
                # More robust pattern matching for the ledger table
                match = re.search(rf"\|.*?{ab['name']}.*?\|", content)
                if match:
                    print(f"LEDGER HIT: Found {ab['name']} in ledger. skipping eval.")
                    in_ledger = True
                else:
                    print(f"LEDGER MISS: {ab['name']} NOT in ledger.")
        
        if not in_ledger:
            eval_log = f"logs/ablations/eval_{ab['name']}.log"
            eval_cmd = [
                sys.executable, "-u", "tools/path_b_eval.py",
                "--name", ab["name"],
                "--ckpt", ckpt_path,
                "--notes", ab["notes"]
            ]
            print(f"Evaluating: {' '.join(eval_cmd)}")
            with open(eval_log, "w") as f:
                subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT)
        
        # --- RELATIONAL LOGIC TEST ---
        logic_log = f"logs/ablations/logic_{ab['name']}.log"
        if not os.path.exists(logic_log) or os.path.getsize(logic_log) == 0:
            print(f"Running Relational Logic Test for {ab['name']}...")
            logic_cmd = [
                sys.executable, "-u", "tools/test_relational_logic.py",
                "--ckpt", ckpt_path,
                "--q1", "2", "--q2", "3", "--seed", "1337"
            ]
            with open(logic_log, "w") as f:
                subprocess.run(logic_cmd, stdout=f, stderr=subprocess.STDOUT)
            
        print(f"Finished {ab['name']}.")

if __name__ == "__main__":
    main()
