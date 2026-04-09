"""Validation script for RAFA using the active GTZAN streaming pipeline."""

import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import load_config
from dataset import RafaGuerrillaDataset
from model import RAFA
from train import circular_mse, compute_delta_phase


def main():
    cfg = load_config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RafaGuerrillaDataset(cfg, split=cfg["data"].get("val_split", "validation"))
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = RAFA(cfg).to(dev)
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    ckpts = sorted(os.listdir(ckpt_dir))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    ckpt = os.path.join(ckpt_dir, ckpts[-1])
    state = torch.load(ckpt, map_location=dev)
    _missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"Checkpoint compatibility warning: unexpected_keys={len(unexpected)}")
    model.eval()

    max_steps = cfg["training"].get("val_steps", 50)
    total = 0.0
    steps = 0

    with torch.no_grad():
        for mags, phs, _labels in loader:
            mags = mags.to(dev)
            phs = phs.to(dev)

            pd, md, _, _ = model(mags, phs, None)
            td = compute_delta_phase(phs)

            loss = circular_mse(pd, td) + F.mse_loss(md, mags)
            total += float(loss.item())
            steps += 1

            if max_steps is not None and steps >= int(max_steps):
                break

    print(f"Validation loss: {total / max(1, steps):.5f} over {steps} step(s)")


if __name__ == "__main__":
    main()
