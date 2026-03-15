import torch
import torch.nn as nn
from model import RAFA
from diffusion_models import RAFADenoiser
from config import load_config

def find_failing_module(name, module):
    try:
        module.to("cuda")
    except Exception as e:
        if "no kernel image" in str(e):
            # If the whole module fails, check its children
            has_failing_child = False
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                if find_failing_module(full_name, child_module):
                    has_failing_child = True
            
            if not has_failing_child:
                print(f"!!! CRITICAL FAIL !!!")
                print(f"Module: {name}")
                print(f"Type:   {type(module)}")
                print(f"Error:  {e}")
                return True
        else:
            # Other errors are secondary
            pass
    return False

def audit_cuda_transfer():
    cfg = load_config()
    print("Initializing model...")
    model = RAFADenoiser(cfg)
    
    print("\n" + "="*60)
    print("CUDA TRANSFER FAULT LOCALIZATION")
    print("="*60)
    
    find_failing_module("", model)

if __name__ == "__main__":
    audit_cuda_transfer()
