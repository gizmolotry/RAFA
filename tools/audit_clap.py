import torch
from transformers import ClapTextModelWithProjection
import os

def audit_clap():
    print("Loading CLAP to check submodules...")
    # Using a dummy model name if not cached, but we just want to see the architecture
    model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
    
    print("\n" + "="*60)
    print("CLAP SUBMODULE AUDIT")
    print("="*60)
    
    found = False
    for name, module in model.named_modules():
        m_type = str(type(module))
        if "gru" in m_type or "LSTM" in m_type or "rnn" in m_type:
            print(f"rnn FOUND IN CLAP: {name:40} | Type: {m_type}")
            found = True
    
    if not found:
        print("No rnns found in CLAP.")

if __name__ == "__main__":
    audit_clap()
