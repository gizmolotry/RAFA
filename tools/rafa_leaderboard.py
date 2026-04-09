import os
import json
import torch
import numpy as np
from config import load_config

def audit_gates():
    cfg = load_config()
    leaderboard = {
        "Stage 1 (Voice)": {"Target": "MSE < 0.05", "Status": "INACTIVE", "Current": None},
        "Stage 2 (Spine)": {"Target": "MSE < 0.1", "Status": "INACTIVE", "Current": None},
        "Stage 3 (Brain)": {"Target": "Tension Corr > 0.4", "Status": "INACTIVE", "Current": None},
        "Stage 4 (Joint)": {"Target": "Success > 0.8", "Status": "INACTIVE", "Current": None}
    }
    
    # Audit Stage 1 (Voice)
    if os.path.exists("logs/stage1_voice.log"):
        leaderboard["Stage 1 (Voice)"]["Status"] = "ACTIVE"
        with open("logs/stage1_voice.log", "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "VOICE LOSS:" in line:
                    try:
                        loss = float(line.split("VOICE LOSS:")[1].strip())
                        leaderboard["Stage 1 (Voice)"]["Current"] = f"{loss:.4f}"
                        if loss < 0.05:
                            leaderboard["Stage 1 (Voice)"]["Status"] = "PASSED"
                        break
                    except: continue

    # Audit Stage 2 (Spine)
    if os.path.exists("logs/stage2_spine.log"):
        leaderboard["Stage 2 (Spine)"]["Status"] = "ACTIVE"
        with open("logs/stage2_spine.log", "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "SPINE LOSS:" in line:
                    try:
                        loss = float(line.split("SPINE LOSS:")[1].strip())
                        leaderboard["Stage 2 (Spine)"]["Current"] = f"{loss:.4f}"
                        if loss < 0.1:
                            leaderboard["Stage 2 (Spine)"]["Status"] = "PASSED"
                        break
                    except: continue

    # Audit Stage 3 (Brain)
    if os.path.exists("logs/master_training.log"):
        leaderboard["Stage 3 (Brain)"]["Status"] = "ACTIVE"
        with open("logs/master_training.log", "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "CORR:" in line:
                    try:
                        corr = float(line.split("CORR:")[1].split("|")[0].strip())
                        leaderboard["Stage 3 (Brain)"]["Current"] = f"{corr:.4f}"
                        if corr > 0.4:
                            leaderboard["Stage 3 (Brain)"]["Status"] = "PASSED"
                        break
                    except: continue
                    
    print("\n" + "="*60)
    print("RAFA DAG PROMOTION GATES (sm_120)")
    print("="*60)
    for stage, info in leaderboard.items():
        print(f"{stage:20} | {info['Status']:8} | Target: {info['Target']:20} | Current: {info['Current']}")

if __name__ == "__main__":
    audit_gates()
