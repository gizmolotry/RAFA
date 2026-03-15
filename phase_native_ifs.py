import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from triton_kernels import ramanujan_summary_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

@dataclass
class PhaseNativeIFSConfig:
    enabled: bool = True
    num_steps: int = 8
    num_maps: int = 4
    delta_max: float = 0.35
    qset: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 12)
    q_weights: tuple[float, ...] = (1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    alpha: float = 1.0
    beta: float = 0.0
    temp: float = 1.0
    hidden_dim: int = 32
    routing_gumbel: bool = False
    routing_tau: float = 1.0
    routing_tau_min: float = 0.5
    routing_tau_decay: float = 1.0
    router_entropy_weight: float = 0.0
    freeze_router: bool = False
    freeze_maps: bool = False
    detach_every: int | None = None
    noise_enabled: bool = False
    noise_sigma0: float = 0.0
    noise_sigma_min: float = 0.0
    mode: str = "full"
    block_size: int = 128
    power_iter_steps: int = 8
    exact_eig_debug: bool = False
    debug_store_a: bool = False
    slow_clock_enabled: bool = True
    slow_pool: int = 16
    slow_hidden_dim: int = 32
    slow_delta_floor: float = 0.35
    slow_delta_ceil: float = 1.0
    time_stride: int = 1
    use_triton: bool = False
    memory_enabled: bool = True
    memory_dim: int = 64
    impedance_enabled: bool = True
    h0_anchor_enabled: bool = False
    h0_anchor_lambda: float = 0.0
    h0_anchor_lambda_min: float = 0.0
    h0_anchor_decay: float = 1.0
    intermediate_consistency_enabled: bool = False
    intermediate_consistency_detach_target: bool = True

def _renorm(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = torch.sqrt((z * z).sum(dim=-1, keepdim=True).clamp_min(eps))
    return z / n

def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)

def _conj(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z[..., 0], -z[..., 1]], dim=-1)

def _exp_i(theta: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

def _circular_mean(delta_k: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    s = (pi * torch.sin(delta_k)).sum(dim=-1)
    c = (pi * torch.cos(delta_k)).sum(dim=-1)
    eps = 1e-12
    mask = (s.abs() < eps) & (c.abs() < eps)
    return torch.atan2(s, c + mask.float() * eps)

class _MapHead(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class SemanticImpedanceHead(nn.Module):
    def __init__(self, num_q: int):
        super().__init__()
        self.num_q = num_q
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, num_q + 2),
        )
    def forward(self, tension: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(tension)
        q_masks = torch.sigmoid(out[..., :self.num_q])
        temp_scale = 0.5 + torch.sigmoid(out[..., self.num_q : self.num_q+1])
        alpha_scale = 0.5 + torch.sigmoid(out[..., self.num_q+1 :])
        return {"q_masks": q_masks, "temp_scale": temp_scale, "alpha_scale": alpha_scale}

# Strategy 2: Functional Stepper (Blackwellsm_120 Safe)
def functional_stepper(x_t, h_prev, w_ih, w_hh, b_ih, b_hh):
    gates_i = F.linear(x_t, w_ih, b_ih)
    gates_h = F.linear(h_prev, w_hh, b_hh)
    i_r, i_z, i_n = gates_i.chunk(3, dim=-1)
    h_r, h_z, h_n = gates_h.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1.0 - z) * n + z * h_prev

from rafa_clutch_transformer_upgraded import BlackwellSafeAttention

class RelationalMemory(nn.Module):
    def __init__(self, feat_dim: int, mem_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = BlackwellSafeAttention(feat_dim, num_heads, batch_first=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x)
        return x + attn_out

class PhaseNativeIFS(nn.Module):
    def __init__(self, q_bins: int, cfg: dict[str, Any] | None = None):
        super().__init__()
        c = cfg or {}
        print(f"DEBUG: PhaseNativeIFS Stealth Overhaul Active")
        self.cfg = PhaseNativeIFSConfig(**{k: v for k, v in c.items() if k in PhaseNativeIFSConfig.__dataclass_fields__})
        self.q_bins = q_bins
        feat_dim = 5
        self.router = nn.Sequential(
            nn.Linear(feat_dim, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.num_maps),
        )
        self.maps = nn.ModuleList([_MapHead(feat_dim, self.cfg.hidden_dim) for _ in range(self.cfg.num_maps)])
        
        # Blackwell sm_120 Fix: Raw parameters instead of nn.module subclass
        hdim = self.cfg.slow_hidden_dim
        self.w_ih = nn.Parameter(torch.empty(3 * hdim, 3))
        self.w_hh = nn.Parameter(torch.empty(3 * hdim, hdim))
        self.b_ih = nn.Parameter(torch.zeros(3 * hdim))
        self.b_hh = nn.Parameter(torch.zeros(3 * hdim))
        nn.init.xavier_uniform_(self.w_ih)
        nn.init.orthogonal_(self.w_hh)

        if self.cfg.memory_enabled:
            self.memory = RelationalMemory(feat_dim=3, mem_dim=self.cfg.memory_dim)
        else:
            self.memory = nn.Identity()
        self.impedance_head = SemanticImpedanceHead(len(self.cfg.qset)) if self.cfg.impedance_enabled else None
        self.slow_proj = nn.Linear(hdim, 3)
        self.register_buffer("qset_t", torch.tensor(self.cfg.qset, dtype=torch.int32))
        self.register_buffer("qw_t", torch.tensor(self.cfg.q_weights, dtype=torch.float32))
        self.w_mod = nn.Parameter(torch.zeros(q_bins, q_bins)) if self.cfg.beta != 0.0 else None

    def _ramanujan_score_from_r(self, r, qset_scale=None):
        ry, rx = r[..., 1], r[..., 0]
        theta = torch.atan2(ry, rx + 1e-12)
        out = torch.zeros_like(theta)
        for qi, (q, w) in enumerate(zip(self.cfg.qset, self.cfg.q_weights)):
            term = float(w) * torch.cos(float(q) * theta)
            if qset_scale is not None:
                s = qset_scale[:, qi]
                while s.dim() < term.dim(): s = s.unsqueeze(-1)
                term = term * s
            out = out + term
        return out

    def _relative_phasor_full(self, z):
        return _mul(z.unsqueeze(2), _conj(z.unsqueeze(1)))

    def build_operator(self, z, alpha_scale=None, temp_scale=None, bias_i=None, qset_scale=None):
        r = self._relative_phasor_full(z)
        s = self._ramanujan_score_from_r(r, qset_scale=qset_scale)   
        cur_a = self.cfg.alpha * (alpha_scale.unsqueeze(-1).unsqueeze(-1) if alpha_scale is not None else 1.0)
        logits = cur_a * s
        if self.w_mod is not None: logits = logits + self.cfg.beta * self.w_mod.unsqueeze(0)
        if bias_i is not None: logits = logits + bias_i.unsqueeze(-1)
        cur_t = max(self.cfg.temp, 1e-6) * (temp_scale.unsqueeze(-1).unsqueeze(-1) if temp_scale is not None else 1.0)
        return torch.softmax(logits / cur_t, dim=-1), s

    def _router_features(self, z, a, s):
        m = (a.unsqueeze(-1) * self._relative_phasor_full(z)).sum(dim=2)
        m_re, m_im = m[..., 0], m[..., 1]
        m_abs = torch.sqrt((m * m).sum(dim=-1).clamp_min(1e-12))
        ent = -(a * torch.log(a.clamp_min(1e-12))).sum(dim=-1)
        diag = torch.diagonal(a, dim1=1, dim2=2)
        return torch.stack([m_re, m_im, m_abs, ent, diag], dim=-1), (a * s).sum(dim=-1)

    def _forward_single(self, z0, alpha_scale=None, temp_scale=None, bias_i=None, delta_scale=None, qset_scale=None, tension=None):
        dev = z0.device
        z = _renorm(z0)
        h0 = z
        imp_alpha, imp_temp, imp_q = 1.0, 1.0, None
        if self.impedance_head and tension is not None:
            io = self.impedance_head(tension.to(dev))
            imp_q, imp_alpha, imp_temp = io["q_masks"], io["alpha_scale"], io["temp_scale"]
        
        for s_idx in range(self.cfg.num_steps):
            ca = (alpha_scale * imp_alpha) if alpha_scale is not None else imp_alpha
            ct = (temp_scale * imp_temp) if temp_scale is not None else imp_temp
            cq = qset_scale * imp_q if (qset_scale is not None and imp_q is not None) else (qset_scale or imp_q)
            a, score = self.build_operator(z, alpha_scale=ca, temp_scale=ct, bias_i=bias_i, qset_scale=cq)
            feats, _ = self._router_features(z, a, score)
            pi = torch.softmax(self.router(feats), dim=-1)
            dk = torch.stack([m(feats) for m in self.maps], dim=-1)
            delta = _circular_mean(dk * self.cfg.delta_max, pi)
            if delta_scale is not None: delta = delta * delta_scale.view(delta.shape[0], 1)
            z = _renorm(_mul(z, _exp_i(delta)))
        return z, {}

    def forward_sequence(self, z_seq, ext_controls=None, tension=None):
        z_seq = _renorm(z_seq)
        bsz, t_len, q, _ = z_seq.shape
        p = max(1, int(self.cfg.slow_pool))
        n_blocks = (t_len + p - 1) // p
        h_slow = z_seq.new_zeros(bsz, self.cfg.slow_hidden_dim)
        z_prev = z_seq[:, 0]
        outs = []
        for b in range(n_blocks):
            t0, t1 = b * p, min(t_len, (b + 1) * p)
            if self.cfg.slow_clock_enabled:
                a_e, s_e = self.build_operator(z_prev)
                f_e, _ = self._router_features(z_prev, a_e, s_e)
                f_v = torch.stack([f_e[..., 2].mean(1), f_e[..., 3].mean(1), z_seq.new_zeros(bsz)], dim=-1).unsqueeze(1)
                if self.cfg.memory_enabled: f_v = self.memory(f_v)
                # Stealth Step:
                h_slow = functional_stepper(f_v[:, 0], h_slow, self.w_ih, self.w_hh, self.b_ih, self.b_hh)
                ctrl = self.slow_proj(h_slow)
                aw = (0.5 + torch.sigmoid(ctrl[:, 0:1])).expand(-1, t1-t0)
                tw = (0.5 + torch.sigmoid(ctrl[:, 1:2])).expand(-1, t1-t0)
                dw = (self.cfg.slow_delta_floor + (self.cfg.slow_delta_ceil - self.cfg.slow_delta_floor) * torch.sigmoid(ctrl[:, 2:3])).expand(-1, t1-t0)
            else:
                aw = tw = dw = z_seq.new_ones(bsz, t1-t0)

            for t_idx in range(t1 - t0):
                t_abs = t0 + t_idx
                zt, _ = self._forward_single(z_seq[:, t_abs], alpha_scale=aw[:, t_idx], temp_scale=tw[:, t_idx], delta_scale=dw[:, t_idx], tension=tension[:, t_abs] if tension is not None else None)
                if t_abs > 0: zt = _renorm(0.1 * zt + 0.9 * z_prev)
                z_prev = zt
                outs.append(zt)
        return torch.stack(outs, dim=1), {}
