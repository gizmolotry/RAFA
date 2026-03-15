import torch
from model import RAFA
from diffusion_models import RAFADenoiser
from config import load_config

def audit_hierarchy():
    cfg = load_config()
    model = RAFADenoiser(cfg)
    
    print("\n" + "="*60)
    print("RECURSIVE MODULE AUDIT")
    print("="*60)
    
    for name, module in model.named_modules():
        m_type = str(type(module))
        if "torch.nn.modules.kernel" in m_type or "solver" in m_type or "LSTM" in m_type:
            print(f"TARGET FOUND: {name:40} | Type: {m_type}")
        elif "Attention" in m_type or "Multihead" in m_type:
            # MultiheadAttention can sometimes have issues on new archs if using fast path
            print(f"ATTN FOUND:   {name:40} | Type: {m_type}")

if __name__ == "__main__":
    audit_hierarchy()
