
import os
import subprocess
import yaml
import sys
import json
import hashlib
import re
import itertools
from pathlib import Path
from datetime import datetime

def get_manifest_hash(config):
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()

def update_config(overrides):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    for key, value in overrides.items():
        # Handle nested keys like "phase_native_ifs.memory_enabled"
        parts = key.split('.')
        target = cfg
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
        
    with open("config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg

def main():
    steps = 1000
    
    # Define the Hypercube Axes (Silent Play Mode)
    grid = {
        "phase_native_ifs.memory_enabled": [True, False],
        "training.losses.use_crystal_rel": [True, False],
        "phase_native_ifs.slow_clock_enabled": [True, False],
        "phase_native_ifs.impedance_enabled": [True, False],
        "training.weights.w_diffusion": [0.0],
        "training.weights.w_latent_align": [1.0]
    }
    
    # Keys for naming the run
    short_keys = {
        "phase_native_ifs.memory_enabled": "mem",
        "training.losses.use_crystal_rel": "ram",
        "phase_native_ifs.slow_clock_enabled": "slow",
        "phase_native_ifs.impedance_enabled": "imp",
        "training.weights.w_diffusion": "diff",
        "training.weights.w_latent_align": "align"
    }

    # Generate all combinations
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total Hypercube Cells: {len(combinations)}")
    
    os.makedirs("logs/hypercube", exist_ok=True)
    msvc_path = r"C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
    os.environ["PATH"] = msvc_path + os.pathsep + os.environ["PATH"]
    os.environ["PYTHONPATH"] = "."
    
    for overrides in combinations:
        # Construct name
        run_name = "pathb_hypercube"
        for k, v in overrides.items():
            run_name += f"_{short_keys[k]}{1 if v else 0}"
            
        print(f"\n" + "="*60 + f"\nCELL: {run_name}\n" + "="*60)
        
        # Check if already in ledger
        if os.path.exists("EVAL_LEDGER.md"):
            with open("EVAL_LEDGER.md", "r") as f:
                if f"| {run_name} |" in f.read():
                    print(f"SKIPPING: {run_name} already in ledger.")
                    continue

        full_cfg = update_config(overrides)
        current_hash = get_manifest_hash(full_cfg)
        ckpt_dir = f"checkpoints_diffusion_{run_name}"
        manifest_path = os.path.join(ckpt_dir, "run_manifest.json")
        
        should_train = True
        ckpt_path = os.path.join(ckpt_dir, f"diff_step{steps}.pt")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                old_manifest = json.load(f)
                if old_manifest.get("manifest_hash") == current_hash and os.path.exists(ckpt_path):
                    print(f"VALID MANIFEST & CKPT: Resuming {run_name}")
                    should_train = False
                else:
                    print(f"MANIFEST MISMATCH OR CKPT MISSING: Retraining {run_name}.")
        
        if should_train:
            os.makedirs(ckpt_dir, exist_ok=True)
            log_path = f"logs/hypercube/train_{run_name}.log"
            train_cmd = [
                sys.executable, "-u", "train_diffusion.py",
                "--model_type", "rafa",
                "--max_steps", str(steps),
                "--ckpt_dir", ckpt_dir
            ]
            print(f"TRAINING: {run_name}")
            with open(log_path, "w", buffering=1) as f:
                subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT)
            
            with open(manifest_path, "w") as f:
                json.dump({
                    "name": run_name,
                    "overrides": overrides,
                    "manifest_hash": current_hash,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

        ckpt_path = os.path.join(ckpt_dir, f"diff_step{steps}.pt")
        if not os.path.exists(ckpt_path):
            files = sorted(list(Path(ckpt_dir).glob("*.pt")), key=os.path.getmtime)
            if files: ckpt_path = str(files[-1])
            else:
                print(f"ERROR: No checkpoint for {run_name}")
                continue
                
        # Evaluate
        eval_log = f"logs/hypercube/eval_{run_name}.log"
        eval_cmd = [
            sys.executable, "-u", "tools/path_b_eval.py",
            "--name", run_name,
            "--ckpt", ckpt_path,
            "--notes", f"Hypercube Cell: {overrides}"
        ]
        print(f"EVALUATING: {run_name}")
        with open(eval_log, "w") as f:
            subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT)
            
        print(f"COMPLETED: {run_name}")

if __name__ == "__main__":
    main()
