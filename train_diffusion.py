from __future__ import annotations

import os
import time
import argparse
import re
import shutil
import json
import random
import hashlib
import subprocess
import math
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

# BLACKWELL WORKAROUND: Disable cuDNN to avoid 'no kernel image' errors on sm_120
torch.backends.cudnn.enabled = False
print("DEBUG: Blackwell Workaround Active (cuDNN disabled)")
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import load_config
from dataset import RafaGuerrillaDataset
from diffusion_models import BaselineDenoiser, RAFADenoiser
from diffusion_utils import make_beta_schedule, extract, phase_to_phasor, q_sample_x0, phasor_normalize, phasor_to_phase, spectral_centroid
from hf_local import resolve_hf_pretrained_path
import rafa_math_tools as rmt


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float((g * g).sum().item())
    return total ** 0.5


def _unpack_batch(batch):
    if len(batch) >= 5:
        return batch[0], batch[1], batch[2], [str(x) for x in batch[3]], batch[4]
    if len(batch) >= 4:
        return batch[0], batch[1], batch[2], [str(x) for x in batch[3]], None
    return batch[0], batch[1], batch[2], None, None


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _dataset_fingerprint(ds: RafaGuerrillaDataset) -> dict[str, object]:
    if bool(getattr(ds, "use_local_files", False)):
        entries = []
        for p in sorted(getattr(ds, "local_files", []), key=lambda x: str(x).lower()):
            pstr = p.as_posix()
            try:
                st = p.stat()
                entries.append(f"{pstr}|{int(st.st_size)}|{int(st.st_mtime_ns)}")
            except OSError:
                entries.append(f"{pstr}|missing")
        h = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
        return {
            "source": "local_wavs",
            "num_files": len(entries),
            "hash": h,
        }
    return {"source": "gtzan", "hash": "none"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--model_type", type=str, default=None, choices=["rafa", "baseline"])
    ap.add_argument("--ckpt_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config()
    dcfg = cfg["diffusion"]
    pcfg = cfg.get("performance", {})
    run_seed = int(dcfg.get("seed", 1337))
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)

    train_raw = RafaGuerrillaDataset(cfg, split="training")
    
    dataset_manifest = _dataset_fingerprint(train_raw)
    git_sha = _git_sha()
    print(f"run_manifest seed={run_seed} git_sha={git_sha} dataset_hash={dataset_manifest['hash'][:12]}")
    
    # LAZY LOADING: num_workers=0 to save system RAM on Windows
    bsz = int(cfg.get("training", {}).get("batch_size", 8))
    loader = DataLoader(train_raw, batch_size=bsz, shuffle=True, num_workers=0, pin_memory=False)

    model_cfg = cfg
    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    model_type = args.model_type if args.model_type is not None else dcfg.get("model_type", "rafa")
    
    if model_type == "baseline":
        # Baseline disabled for Blackwell compatibility
        # model = BaselineDenoiser(freq_bins).to(dev)
        raise NotImplementedError("BaselineDenoiser is currently disabled for sm_120 compatibility.")
    else:
        import phase_native_ifs
        print(f"DEBUG: phase_native_ifs location: {phase_native_ifs.__file__}")
        # BLACKWELL REDIRECT
        from blackwell_model import RAFA
        model = RAFADenoiser(model_cfg).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=float(dcfg.get("lr", 2e-4)))
    timesteps = int(dcfg.get("timesteps", 50))
    betas = make_beta_schedule(timesteps, float(dcfg.get("beta_start", 1e-4)), float(dcfg.get("beta_end", 0.02)), dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    grad_clip = float(cfg.get("training", {}).get("grad_clip", 0.0) or 0.0)
    ckpt_dir = args.ckpt_dir or dcfg.get("checkpoint_dir", "checkpoints_diffusion")
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    epochs = int(args.epochs or dcfg.get("train_epochs", 10))
    max_steps = args.max_steps
    
    print(f"diffusion_train start model_type={model_type} epochs={epochs} bsz={bsz}")
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        n = 0
        t0 = time.time()
        for i, batch in enumerate(loader, start=1):
            mag, phase, _lbl, _prompt, gt_tension = _unpack_batch(batch)
            mag = mag.to(dev, non_blocking=True)
            phase = phase.to(dev, non_blocking=True)
            if gt_tension is not None: gt_tension = gt_tension.to(dev, non_blocking=True)
            
            x0_mag = torch.log1p(mag)
            x0_z = phase_to_phasor(phase)
            t = torch.randint(0, timesteps, (mag.size(0),), device=dev)
            xt_mag, xt_z, eps_mag, eps_z = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)

            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                pred_mag, pred_z, p_ext, _ = model(xt_mag, xt_z, t, semantic_controls={"semantic_tension": gt_tension} if gt_tension is not None else None)
                ab_t = extract(alphas_cumprod, t, x0_mag.shape)
                sqrt_ab_t, sqrt_omab_t = ab_t.sqrt(), (1.0 - ab_t).sqrt().clamp_min(1e-8)
                eps_pred_mag = (xt_mag - sqrt_ab_t * pred_mag) / sqrt_omab_t
                eps_pred_z = (xt_z - sqrt_ab_t.unsqueeze(-1) * pred_z) / sqrt_omab_t.unsqueeze(-1)

                loss_mag = (eps_pred_mag - eps_mag).pow(2).mean()
                loss_phase = (eps_pred_z - eps_z).pow(2).mean()

                tw = cfg.get("training", {}).get("weights", {})
                l_physics = pred_mag.new_zeros(())
                
                if "phase_state" in p_ext and gt_tension is not None:
                    zf = p_ext["phase_state"]
                    h_energy, i_energy = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
                    actual_ratio = i_energy / (h_energy + i_energy).clamp_min(1e-8)
                    
                    if actual_ratio.shape != gt_tension.shape:
                        actual_ratio = F.interpolate(actual_ratio.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
                    
                    l_align = F.mse_loss(actual_ratio, gt_tension)
                    l_physics = l_physics + tw.get("w_latent_align", 1.0) * l_align

                w_diff = float(tw.get("w_diffusion", 1.0))
                loss = w_diff * (loss_mag + loss_phase) + l_physics

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            run_loss += float(loss.detach().item())
            n += 1
            global_step += 1
            if global_step % 20 == 0:
                print(f"step={global_step} loss={run_loss/n:.5f} mag={loss_mag.item():.5f} phase={loss_phase.item():.5f}", flush=True)
            if max_steps and global_step >= max_steps: break
        if max_steps and global_step >= max_steps: break

if __name__ == "__main__":
    main()
