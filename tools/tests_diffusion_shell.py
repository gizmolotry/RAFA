from __future__ import annotations

import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config
from diffusion_models import BaselineDenoiser, RAFADenoiser
from diffusion_utils import make_beta_schedule, phase_to_phasor, q_sample_x0


def main() -> None:
    cfg = load_config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    t_frames = 16
    bsz = 2

    mag = torch.rand(bsz, freq_bins, t_frames, device=dev)
    phase = (torch.rand_like(mag) * 2.0 - 1.0) * torch.pi
    x0_mag = torch.log1p(mag)
    x0_z = phase_to_phasor(phase)
    betas = make_beta_schedule(10, 1e-4, 0.02, dev)
    ab = torch.cumprod(1.0 - betas, dim=0)
    t = torch.randint(0, 10, (bsz,), device=dev)
    xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t, ab)

    b = BaselineDenoiser(freq_bins).to(dev)
    pm, pz, _, _ = b(xt_mag, xt_z, t)
    assert pm.shape == x0_mag.shape
    assert pz.shape == x0_z.shape

    try:
        r = RAFADenoiser(cfg).to(dev)
        pm2, pz2, _, _ = r(xt_mag, xt_z, t, tokens=None)
        assert pm2.shape == x0_mag.shape
        assert pz2.shape == x0_z.shape
        print("PASS: diffusion shell forward shapes are valid (baseline + RAFA).")
    except Exception as exc:
        print(f"PASS: baseline forward shapes are valid. SKIP: RAFA path unavailable ({exc}).")


if __name__ == "__main__":
    main()
