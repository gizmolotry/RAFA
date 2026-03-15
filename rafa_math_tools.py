"""
Utility functions implementing various number-theoretic
and rhythm-aware operations used in RAFA upgrades.

This module provides:
 - Ramanujan sums and rhythm-strength detectors.
 - Periodicity‑weighted circular loss (R1).
 - Modular phase embeddings and projection consistency (R2).
 - Adaptive circular likelihood via a von Mises distribution (R5).
 - Truncated approximations to classical functions such as the Dedekind
   eta function, Ramanujan's 1/π series, mock theta functions,
   Eisenstein series and the Ramanujan τ-function. These are provided
   for optional use in gating or priors.

Where possible the functions operate on NumPy arrays for signal
processing, and on PyTorch tensors for differentiable losses.
"""
import numpy as np
import torch
import torch.nn.functional as F
from math import gcd

# Basic angular operations
def wrap_angle_np(a: np.ndarray) -> np.ndarray:
    """Wrap an array of angles to the interval (-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap a PyTorch tensor of angles to the interval (-π, π]."""
    return (a + torch.pi) % (2 * torch.pi) - torch.pi

def circular_loss(phi_pred: torch.Tensor, phi_true: torch.Tensor) -> torch.Tensor:
    """Basic circular loss: 1 - cos(delta) for wrapped phase difference."""
    delta = wrap_angle(phi_pred - phi_true)
    return 1.0 - torch.cos(delta)

def von_mises_nll(phi_pred: torch.Tensor, phi_true: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Von Mises negative log-likelihood up to a constant."""
    delta = wrap_angle(phi_pred - phi_true)
    return -kappa * torch.cos(delta)

# Ramanujan sums and rhythm detectors
def ramanujan_cq(q: int, L: int) -> np.ndarray:
    n = np.arange(L, dtype=np.float64)
    ks = [k for k in range(1, q + 1) if gcd(k, q) == 1]
    cq = np.zeros(L, dtype=np.complex128)
    for k in ks:
        cq += np.exp(2j * np.pi * k * n / q)
    return cq

def sliding_corr_strength(x: np.ndarray, template: np.ndarray, step: int = 1, norm: bool = True) -> np.ndarray:
    L = len(template)
    if L > len(x):
        raise ValueError("Template longer than signal.")
    strengths = []
    tnorm = np.linalg.norm(template)
    for start in range(0, len(x) - L + 1, step):
        w = x[start:start + L]
        c = np.vdot(template, w)
        if norm:
            denom = tnorm * (np.linalg.norm(w) + 1e-9)
            strengths.append(np.abs(c) / (denom + 1e-12))
        else:
            strengths.append(np.abs(c))
    strengths = np.array(strengths)
    pad = len(x) - len(strengths)
    if pad > 0:
        strengths = np.pad(strengths, (pad, 0))
    return strengths

def ramanujan_rhythm_strength(signal: np.ndarray, qs: tuple[int, ...] = (2,3,4,5,6,8), window: int = 256, step: int = 1) -> dict:
    per_q = {}
    for q in qs:
        template = ramanujan_cq(q, window)
        per_q[q] = sliding_corr_strength(signal, template, step=step, norm=True)
    stacked = np.stack([per_q[q] for q in qs], axis=0)
    envelope = np.max(stacked, axis=0)
    return {'per_q': per_q, 'max': envelope}

# Loss functions for RAFA upgrades
def ramanujan_weighted_circular_loss(phi_pred: np.ndarray, phi_true: np.ndarray, signal: np.ndarray,
                                     lam: float = 1.0, qs: tuple[int, ...] = (2,3,4,5,6,8),
                                     window: int = 256, step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = 1.0 - np.cos(wrap_angle_np(phi_pred - phi_true))
    R = ramanujan_rhythm_strength(signal, qs=qs, window=window, step=step)['max']
    L = min(len(base), len(R))
    base = base[:L]; R = R[:L]
    weighted = (1.0 + lam * R) * base
    return weighted, base, R

def modular_phase_embed_np(phi: np.ndarray, qs: tuple[int, ...] = (3,4,5)) -> np.ndarray:
    feats = [np.cos(q * phi) for q in qs] + [np.sin(q * phi) for q in qs]
    return np.stack(feats, axis=-1)

def modular_phase_embed_loss(phi_pred: torch.Tensor, phi_true: torch.Tensor, qs: tuple[int, ...] = (3,4,5),
                             reduction: str = "mean") -> torch.Tensor:
    embed_p = torch.cat([torch.cos(q * phi_pred) for q in qs] +
                        [torch.sin(q * phi_pred) for q in qs], dim=-1)
    embed_t = torch.cat([torch.cos(q * phi_true) for q in qs] +
                        [torch.sin(q * phi_true) for q in qs], dim=-1)
    mse = (embed_p - embed_t)**2
    return mse.mean() if reduction == "mean" else mse.sum()

def adaptive_von_mises_loss(phi_pred: np.ndarray, phi_true: np.ndarray, signal: np.ndarray,
                            kappa0: float = 1.0, alpha: float = 1.0,
                            qs: tuple[int, ...] = (2,3,4,5,6,8), window: int = 256, step: int = 1) -> float:
    delta = wrap_angle_np(phi_pred - phi_true)
    R = ramanujan_rhythm_strength(signal, qs=qs, window=window, step=step)['max']
    L = min(len(delta), len(R))
    delta = delta[:L]; R = R[:L]
    kappa = kappa0 + alpha * R
    return float((-kappa * np.cos(delta)).mean())

# Truncated classical functions
def dedekind_eta_q(τ: float, N: int = 50) -> complex:
    q = np.exp(2j * np.pi * τ)
    prod = np.prod([(1 - q**n) for n in range(1, N+1)])
    return q**(1/24) * prod

# ... (Eisenstein functions, tau function, etc., omitted for brevity)

def tau_gate_mask(freqs: np.ndarray, eps: float = 0.1) -> np.ndarray:
    n = len(freqs)
    M = np.zeros((n, n), dtype=np.float32)
    max_tau = max(abs(ramanujan_tau(i)) for i in range(1, n+1)) or 1
    for i in range(n):
        for j in range(n):
            g = int(np.gcd(int(freqs[i]), int(freqs[j])))
            tau_val = ramanujan_tau(g)
            weight = abs(tau_val) / max_tau
            M[i, j] = 1.0 if weight > eps else 0.0
    return M

# RAFA-PC v3.0 extensions: rational and harmonic priors
def soft_rational_prior(omegas: torch.Tensor, rationals: list[tuple[int,int]],
                        temp: float = 50.0, pow_r: float = 2.0) -> torch.Tensor:
    """
    Vectorized version of the soft rational prior loss.
    Encourages frequency ratios to align with simple integer ratios.
    """
    Q = omegas.size(0)
    if Q < 2 or not rationals:
        return omegas.new_zeros(())
    
    # Create a matrix of all pairwise ratios
    omegas_col = omegas.unsqueeze(1)
    omegas_row = omegas.unsqueeze(0)
    # Clamp omegas to avoid division by zero and exploding gradients
    omegas_safe = torch.clamp(omegas_row, min=1e-4)
    ratio_matrix = torch.abs(omegas_col / omegas_safe) # Shape: (Q, Q)
    
    # Create tensors for target ratios and their weights
    device = omegas.device
    targets = torch.tensor([p / r for p, r in rationals], device=device) # Shape: (num_rationals,)
    r_weights = torch.tensor([r**pow_r for p, r in rationals], device=device) # Shape: (num_rationals,)

    # Expand dimensions for broadcasting
    # ratio_matrix -> (Q, Q, 1)
    # targets -> (1, 1, num_rationals)
    # r_weights -> (1, 1, num_rationals)
    dists = torch.abs(ratio_matrix.unsqueeze(-1) - targets.view(1, 1, -1)) * r_weights.view(1, 1, -1)
    
    # Compute softmax weights and the weighted loss for all pairs
    weights = torch.softmax(-temp * dists, dim=-1)
    loss_matrix = (weights * dists).sum(dim=-1) # Shape: (Q, Q)
    
    # Mean over unique pairs for scale stability across different Q.
    tri = torch.triu(loss_matrix, diagonal=1)
    pair_count = max(1, (Q * (Q - 1)) // 2)
    return tri.sum() / float(pair_count)

def harmonic_coupling_loss(delta_phi: torch.Tensor, rationals: list[tuple[int,int]],
                           lam: float = 1.0) -> torch.Tensor:
    Q, T = delta_phi.shape
    if Q < 2 or not rationals:
        return delta_phi.new_zeros(())
    total = delta_phi.new_zeros(())
    pair_count = 0
    for i in range(Q):
        for j in range(i + 1, Q):
            best_val = None
            for (p, r) in rationals:
                r_float = float(r)
                p_float = float(p)
                scaled = (p_float / r_float) * delta_phi[j]
                # Keep the coupling loss circular-safe across branch cuts at +/-pi.
                err = wrap_angle(delta_phi[i] - scaled)
                val = (err**2).mean()
                if best_val is None or val < best_val:
                    best_val = val
            if best_val is not None:
                total = total + best_val
                pair_count += 1
    return lam * (total / float(max(1, pair_count)))


def phase_to_phasor(phi: torch.Tensor) -> torch.Tensor:
    """Convert phase angles to unit phasors [re, im]."""
    return torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)


def phasor_normalize(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize phasor tensor with last dim [re, im] to unit magnitude."""
    mag = torch.sqrt((z**2).sum(dim=-1, keepdim=True).clamp_min(eps))
    return z / mag


def phasor_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex multiply for [re, im] tensors."""
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def phasor_conj(z: torch.Tensor) -> torch.Tensor:
    """Complex conjugate for [re, im] tensors."""
    return torch.stack([z[..., 0], -z[..., 1]], dim=-1)


def circular_mean_delta(delta_phi_k: torch.Tensor, pi_k: torch.Tensor) -> torch.Tensor:
    """
    Circular mean over map outputs.
    Expects matching shapes with map axis in the last dimension.
    """
    sx = (pi_k * torch.sin(delta_phi_k)).sum(dim=-1)
    cx = (pi_k * torch.cos(delta_phi_k)).sum(dim=-1)
    return torch.atan2(sx, cx)


def phasor_apply_delta(z: torch.Tensor, delta_phi: torch.Tensor) -> torch.Tensor:
    """z <- z * exp(i * delta_phi), with renormalization."""
    u = phase_to_phasor(delta_phi)
    return phasor_normalize(phasor_mul(z, u))


def ramanujan_relative_score_phasor(
    z: torch.Tensor,
    qs: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 12),
    q_weights: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Major-arc coherence score from relative phasors only.
    z shape: (Q, T, 2), unit phasors.
    """
    if z.dim() != 3 or z.size(-1) != 2:
        raise ValueError("z must have shape (Q, T, 2)")

    zr = z[..., 0]
    zi = z[..., 1]

    # r_ij = z_i * conj(z_j)
    re = zr.unsqueeze(1) * zr.unsqueeze(0) + zi.unsqueeze(1) * zi.unsqueeze(0)
    im = zi.unsqueeze(1) * zr.unsqueeze(0) - zr.unsqueeze(1) * zi.unsqueeze(0)
    rel_theta = torch.atan2(im, re)  # (Q, Q, T)

    device = z.device
    dtype = z.dtype
    if q_weights is None:
        q_weights_t = torch.ones(len(qs), dtype=dtype, device=device)
    else:
        q_weights_t = q_weights.to(device=device, dtype=dtype)
        if q_weights_t.numel() != len(qs):
            raise ValueError("q_weights length must match qs length")

    total = z.new_zeros(z.size(0), z.size(0))
    for idx, q in enumerate(qs):
        comp = torch.cos(float(q) * rel_theta).mean(dim=-1)
        if gamma != 1.0:
            comp = torch.sign(comp) * torch.abs(comp).pow(gamma)
        total = total + q_weights_t[idx] * comp
    return total


def ramanujan_relative_score(
    phase_state: torch.Tensor,
    qs: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 12),
    q_weights: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Major-arc coherence score built from relative phase only.

    Args:
        phase_state: Tensor of shape (Q, T), phase trajectories for Q bins.
        qs: Small-denominator winding set.
        q_weights: Optional per-q weights, shape (len(qs),).
        gamma: Nonlinearity for sharper lock preference.

    Returns:
        Score matrix S of shape (Q, Q), where larger values mean stronger
        rational relative-phase coherence.
    """
    if phase_state.dim() != 2:
        raise ValueError("phase_state must have shape (Q, T)")

    z = phase_to_phasor(phase_state)  # (Q, T, 2)
    return ramanujan_relative_score_phasor(z, qs=qs, q_weights=q_weights, gamma=gamma)


def crystal_operator_from_score(score: torch.Tensor, alpha: float = 1.0, temp: float = 1.0) -> torch.Tensor:
    """
    Build a transport operator from coherence scores.
    A_ij ~ softmax_j(alpha * S_ij / temp)
    """
    logits = (alpha * score) / max(float(temp), 1e-6)
    return torch.softmax(logits, dim=-1)


def ramanujan_crystal_loss(
    phase_state: torch.Tensor,
    qs: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 12),
    q_weights: torch.Tensor | None = None,
    gamma: float = 1.0,
    alpha: float = 1.0,
    temp: float = 1.0,
) -> torch.Tensor:
    """
    L_crystal = -mean(A * S), with S from relative-phase major-arc scoring.
    """
    score = ramanujan_relative_score(phase_state, qs=qs, q_weights=q_weights, gamma=gamma)
    op = crystal_operator_from_score(score, alpha=alpha, temp=temp)
    return -(op * score).mean()


def cauchy_riemann_stft_loss(mag: torch.Tensor, phase: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Enforces Holomorphic Consistency (Cauchy-Riemann conditions) in the STFT domain.
    For an analytic signal, the log-magnitude and phase are conjugate harmonic functions.
    
    CR Equations (Discrete approx):
    d(phi)/dt  approx  -d(ln M)/dw
    d(phi)/dw  approx   d(ln M)/dt
    """
    # Ensure shapes are [B, F, T]
    if mag.dim() == 4: mag = mag.squeeze(1)
    if phase.dim() == 4: phase = phase.squeeze(1)
    
    log_mag = torch.log(mag + eps)
    
    # Time derivatives
    dphi_dt = wrap_angle(phase[..., 1:] - phase[..., :-1])
    dlm_dt = log_mag[..., 1:] - log_mag[..., :-1]
    
    # Freq derivatives
    dphi_dw = wrap_angle(phase[:, 1:, :] - phase[:, :-1, :])
    dlm_dw = log_mag[:, 1:, :] - log_mag[:, :-1, :]
    
    # Align dimensions for comparison (crop to valid overlap)
    # dphi_dt is [B, F, T-1], dlm_dw is [B, F-1, T]
    # We compare dphi_dt[:, :-1, :] with -dlm_dw[:, :, :-1]
    
    # 1. d(phi)/dt = -d(ln M)/dw
    term1_lhs = dphi_dt[:, :-1, :]
    term1_rhs = -dlm_dw[..., :-1]
    loss1 = (term1_lhs - term1_rhs).pow(2).mean()
    
    # 2. d(phi)/dw = d(ln M)/dt
    term2_lhs = dphi_dw[..., :-1]
    term2_rhs = dlm_dt[:, :-1, :]
    loss2 = (term2_lhs - term2_rhs).pow(2).mean()
    
    return 0.5 * (loss1 + loss2)


def geodesic_phase_upsample(phase: torch.Tensor, scale_factor: float, mode: str = 'linear') -> torch.Tensor:
    """
    Upsamples a phase field respecting the circular geometry (Geodesic Interpolation).
    Instead of linearly interpolating angles (which breaks at +/- pi), 
    we interpolate the unitary phasors (cos, sin) or slerp.
    """
    if scale_factor == 1:
        return phase
        
    z = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1) # [B, C, T, 2]
    B, C, T, _ = z.shape
    
    # Permute for grid_sample or interpolate: [B, 2, C, T] -> treat C as H, T as W
    z_perm = z.permute(0, 3, 1, 2) 
    
    # Upsample the phasor vectors.
    # NOTE: z_perm is 4D ([B,2,H,W]); use bilinear interpolation for 2D fields.
    interp_mode = "bilinear" if z_perm.dim() == 4 and mode == "linear" else mode
    z_up = F.interpolate(z_perm, scale_factor=(1.0, scale_factor), mode=interp_mode, align_corners=True)
    
    # Renormalize (project back to circle)
    z_up = z_up / (torch.norm(z_up, dim=1, keepdim=True) + 1e-8)
    
    # Convert back to phase
    phase_up = torch.atan2(z_up[:, 1, ...], z_up[:, 0, ...])
    
    return phase_up


def harmonic_inharmonic_energy(z, rat_ratios, harmonic_qs=(2, 3, 4, 6, 8, 12)):
    # Expected z: [B, F, T, 2] or [B, T, F, 2]
    # We will detect based on common RAFA shapes (F is usually 129, 257 etc)
    if z.size(1) > z.size(2) and z.size(1) > 100: # Likely [B, F, T, 2]
        z = z.transpose(1, 2) # Now [B, T, F, 2]
    
    B, T, F, _ = z.shape
    # relative phasor
    zi = z.unsqueeze(3) # [B, T, F, 1, 2]
    zj = z.unsqueeze(2) # [B, T, 1, F, 2]
    zj_conj = torch.stack([zj[..., 0], -zj[..., 1]], dim=-1)
    
    r_re = zi[..., 0] * zj_conj[..., 0] - zi[..., 1] * zj_conj[..., 1]
    r_im = zi[..., 0] * zj_conj[..., 1] + zi[..., 1] * zj_conj[..., 0]
    theta = torch.atan2(r_im, r_re) # [B, T, F, F]
    
    all_qs = set()
    for r in rat_ratios:
        all_qs.add(int(r[0]))
        all_qs.add(int(r[1]))
    
    h_energy = torch.zeros((B, T), device=z.device)
    i_energy = torch.zeros((B, T), device=z.device)
    
    for q in all_qs:
        coh = torch.cos(float(q) * theta).mean(dim=(2, 3)) # [B, T]
        if q in harmonic_qs:
            h_energy = h_energy + coh
        else:
            i_energy = i_energy + coh
            
    return h_energy, i_energy
