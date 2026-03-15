from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from config import load_config
from dataset import RafaGuerrillaDataset
from diffusion_models import BaselineDenoiser
from diffusion_utils import phase_to_phasor


def main() -> None:
    cfg = load_config()
    pcfg = cfg.get("performance", {})
    dcfg = cfg.get("diffusion", {})
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RafaGuerrillaDataset(cfg)
    nw = int(pcfg.get("num_workers", 0))
    loader = DataLoader(
        ds,
        batch_size=int(dcfg.get("train_batch_size", 2)),
        shuffle=True,
        num_workers=max(0, nw),
        pin_memory=bool(pcfg.get("pin_memory", False)),
        persistent_workers=bool(pcfg.get("persistent_workers", False)) and nw > 0,
    )
    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    model = BaselineDenoiser(freq_bins).to(dev)
    if bool(pcfg.get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(pcfg.get("use_amp", True)) and dev.type == "cuda")

    batch = next(iter(loader))
    mag, phase, _ = batch[:3]
    mag = mag.to(dev)
    z = phase_to_phasor(phase.to(dev))
    t = torch.randint(0, int(dcfg.get("timesteps", 50)), (mag.size(0),), device=dev)

    torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
    t0 = time.time()
    with torch.autocast(device_type=dev.type, dtype=torch.float16 if dev.type == "cuda" else torch.float32, enabled=bool(pcfg.get("use_amp", True)) and dev.type == "cuda"):
        pm, pz = model(torch.log1p(mag), z, t)
        loss = pm.mean() + pz[..., 0].mean()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    dt = time.time() - t0
    frames = mag.numel()
    fps = frames / max(dt, 1e-6)
    mem = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if dev.type == "cuda" else 0.0
    print(f"step_time_sec={dt:.4f} frames_per_sec={fps:.2f} gpu_mem_mb={mem:.1f}")


if __name__ == "__main__":
    main()
