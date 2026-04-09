"""Minimal diffusion utilities for RAFA mag+phase generation."""

from __future__ import annotations

import torch


def make_beta_schedule(timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=torch.float32)


def extract(v: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = v.gather(0, t.long()).view(-1, *([1] * (len(shape) - 1)))
    return out


def phase_to_phasor(phase: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)


def phasor_to_phase(z: torch.Tensor) -> torch.Tensor:
    return torch.atan2(z[..., 1], z[..., 0])


def phasor_normalize(z: torch.Tensor) -> torch.Tensor:
    return z / torch.sqrt((z * z).sum(dim=-1, keepdim=True).clamp_min(1e-8))


def q_sample_x0(
    x0_mag: torch.Tensor,
    x0_z: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noise_mag = torch.randn_like(x0_mag)
    noise_z = torch.randn_like(x0_z)
    sqrt_ab = extract(alphas_cumprod.sqrt(), t, x0_mag.shape)
    sqrt_omab = extract((1.0 - alphas_cumprod).sqrt(), t, x0_mag.shape)
    xt_mag = sqrt_ab * x0_mag + sqrt_omab * noise_mag
    xt_z = phasor_normalize(sqrt_ab.unsqueeze(-1) * x0_z + sqrt_omab.unsqueeze(-1) * noise_z)
    return xt_mag, xt_z, noise_mag, noise_z


def spectral_centroid(mag: torch.Tensor, sample_rate: int, n_fft: int) -> torch.Tensor:
    # mag: [B,F,T] -> returns [B,T]
    freqs = torch.linspace(0.0, sample_rate / 2.0, mag.size(1), device=mag.device).view(1, -1, 1)
    num = (mag * freqs).sum(dim=1)
    den = mag.sum(dim=1).clamp_min(1e-8)
    return num / den
