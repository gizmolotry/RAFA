import torch
from model import RAFA
from diffusion_models import RAFADenoiser
from config import load_config

def audit():
    cfg = load_config()
    rafa = RAFA(cfg)
    denoiser = RAFADenoiser(cfg)
    
    print("\n" + "="*50)
    print("RAFA CORE AUDIT")
    print("="*50)
    for name, child in rafa.named_children():
        train_p = sum(p.numel() for p in child.parameters() if p.requires_grad)
        frozen_p = sum(p.numel() for p in child.parameters() if not p.requires_grad)
        total = train_p + frozen_p
        if total > 0:
            print(f"{name:20} | Trainable: {train_p/1e6:6.2f}M | Frozen: {frozen_p/1e6:6.2f}M")
            # Sub-audit for phase_solver
            if name == "phase_solver":
                for subname, subchild in child.named_children():
                    s_train = sum(p.numel() for p in subchild.parameters() if p.requires_grad)
                    s_frozen = sum(p.numel() for p in subchild.parameters() if not p.requires_grad)
                    if (s_train + s_frozen) > 0:
                        print(f"  > {subname:18} | Trainable: {s_train/1e3:6.1f}K | Frozen: {s_frozen/1e3:6.1f}K")

    print("\n" + "="*50)
    print("DENOISER WRAPPER AUDIT")
    print("="*50)
    for name, child in denoiser.named_children():
        if name == "rafa": continue # already audited
        train_p = sum(p.numel() for p in child.parameters() if p.requires_grad)
        frozen_p = sum(p.numel() for p in child.parameters() if not p.requires_grad)
        print(f"{name:20} | Trainable: {train_p/1e6:6.2f}M | Frozen: {frozen_p/1e6:6.2f}M")

if __name__ == "__main__":
    audit()
