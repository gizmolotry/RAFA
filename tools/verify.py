from __future__ import annotations

import pathlib
import sys

import torch
from transformers import CLIPTokenizer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config
from model import RAFA


def main() -> None:
    cfg = load_config()
    cfg["model"]["text_conditioned"] = True
    model = RAFA(cfg)
    tok = CLIPTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    tokens = tok(["a crackling fireplace"], return_tensors="pt")

    n_fft = cfg["data"]["stft"]["n_fft"]
    freq_bins = n_fft // 2 + 1
    mags = torch.randn(1, freq_bins, 50)
    phs = torch.randn_like(mags)

    final_phase, final_mag, _, _ = model(mags, phs, tokens)
    print("Promptable forward pass ok:", final_phase.shape, final_mag.shape)


if __name__ == "__main__":
    main()
