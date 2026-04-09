import torch
import torch.nn as nn
from model import RAFA
from diffusion_models import RAFADenoiser
from config import load_config

def total_recall():
    cfg = load_config()
    print("Initializing model...")
    model = RAFADenoiser(cfg)
    
    print("\n" + "="*60)
    print("rnn BASE CLASS SEARCH")
    print("="*60)
    
    # Search for anything inheriting from rnnBase (the common parent of gru/LSTM)
    from torch.nn.modules.rnn import rnnBase
    
    found_any = False
    for name, module in model.named_modules():
        if isinstance(module, rnnBase):
            print(f"rnn FOUND: {name:40} | Type: {type(module)}")
            found_any = True
            
    if not found_any:
        print("No rnnBase instances found in model.named_modules().")
        print("Checking for hidden attributes...")
        for name, module in model.named_modules():
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    if isinstance(attr, rnnBase):
                        print(f"HIDDEN rnn: {name}.{attr_name:30} | Type: {type(attr)}")
                        found_any = True
                except:
                    continue

    if not found_any:
        print("Model is clean of standard rnns. Searching for MultiheadAttention (sm_120 suspect)...")
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                print(f"ATTN FOUND: {name:40}")

if __name__ == "__main__":
    total_recall()
