
import torch
import time
from model import RAFA
from config import load_config

def profile_rafa():
    cfg = load_config()
    # Force small batch for profiling
    cfg["training"]["batch_size"] = 1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dev.type != "cuda":
        print("CUDA not available. Profiling on CPU is not useful for Triton planning.")
        return

    model = RAFA(cfg).to(dev)
    
    # Mock data: [B, F, T]
    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    t_frames = 128
    mag = torch.randn(1, freq_bins, t_frames).to(dev).abs()
    phase = torch.randn(1, freq_bins, t_frames).to(dev)

    # Warmup
    for _ in range(5):
        _ = model(mag, phase)
    
    torch.cuda.synchronize()
    
    # Profile with detailed breakdown
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("rafa_forward"):
            final_phase, final_mag, p_ext, m_ext = model(mag, phase)
            
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    profile_rafa()
