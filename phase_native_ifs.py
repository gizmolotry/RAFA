import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    return torch.atan2(s, c)


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


class PhaseNativeIFS(nn.Module):
    def __init__(self, q_bins: int, cfg: dict[str, Any] | None = None):
        super().__init__()
        c = cfg or {}
        self.cfg = PhaseNativeIFSConfig(
            enabled=bool(c.get("enabled", True)),
            num_steps=int(c.get("num_steps", 8)),
            num_maps=int(c.get("num_maps", 4)),
            delta_max=float(c.get("delta_max", 0.35)),
            qset=tuple(int(x) for x in c.get("qset", [2, 3, 4, 5, 6, 8, 12])),
            q_weights=tuple(float(x) for x in c.get("q_weights", [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])),
            alpha=float(c.get("alpha", 1.0)),
            beta=float(c.get("beta", 0.0)),
            temp=float(c.get("temp", 1.0)),
            hidden_dim=int(c.get("hidden_dim", 32)),
            routing_gumbel=bool(c.get("routing_gumbel", False)),
            routing_tau=float(c.get("routing_tau", 1.0)),
            routing_tau_min=float(c.get("routing_tau_min", 0.5)),
            routing_tau_decay=float(c.get("routing_tau_decay", 1.0)),
            router_entropy_weight=float(c.get("router_entropy_weight", 0.0)),
            freeze_router=bool(c.get("freeze_router", False)),
            freeze_maps=bool(c.get("freeze_maps", False)),
            detach_every=c.get("detach_every", None),
            noise_enabled=bool(c.get("noise_enabled", False)),
            noise_sigma0=float(c.get("noise_sigma0", 0.0)),
            noise_sigma_min=float(c.get("noise_sigma_min", 0.0)),
            mode=str(c.get("mode", "full")),
            block_size=int(c.get("block_size", 128)),
            power_iter_steps=int(c.get("power_iter_steps", 8)),
            exact_eig_debug=bool(c.get("exact_eig_debug", False)),
            debug_store_a=bool(c.get("debug_store_a", False)),
            slow_clock_enabled=bool(c.get("slow_clock_enabled", True)),
            slow_pool=int(c.get("slow_pool", 16)),
            slow_hidden_dim=int(c.get("slow_hidden_dim", 32)),
            slow_delta_floor=float(c.get("slow_delta_floor", 0.35)),
            slow_delta_ceil=float(c.get("slow_delta_ceil", 1.0)),
            time_stride=int(c.get("time_stride", 1)),
        )
        self.q_bins = q_bins

        feat_dim = 5
        self.router = nn.Sequential(
            nn.Linear(feat_dim, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.num_maps),
        )
        self.maps = nn.ModuleList([_MapHead(feat_dim, self.cfg.hidden_dim) for _ in range(self.cfg.num_maps)])
        self.slow_gru = nn.GRU(input_size=3, hidden_size=self.cfg.slow_hidden_dim, batch_first=True)
        self.slow_proj = nn.Linear(self.cfg.slow_hidden_dim, 3)

        if self.cfg.beta != 0.0:
            self.w_mod = nn.Parameter(torch.zeros(q_bins, q_bins))
        else:
            self.w_mod = None

        if self.cfg.freeze_router:
            for p in self.router.parameters():
                p.requires_grad = False
        if self.cfg.freeze_maps:
            for m in self.maps:
                for p in m.parameters():
                    p.requires_grad = False

    def _ramanujan_score_from_r(self, r: torch.Tensor) -> torch.Tensor:
        out = r.new_zeros(r.shape[0], r.shape[1], r.shape[2])
        for q, w in zip(self.cfg.qset, self.cfg.q_weights):
            rq = r
            for _ in range(1, q):
                rq = _mul(rq, r)
                rq = _renorm(rq)
            out = out + float(w) * rq[..., 0]
        return out

    def _relative_phasor_full(self, z: torch.Tensor) -> torch.Tensor:
        return _mul(z.unsqueeze(2), _conj(z.unsqueeze(1)))

    def _score_block(self, z: torch.Tensor, j0: int, j1: int) -> torch.Tensor:
        r = _mul(z.unsqueeze(2), _conj(z[:, j0:j1, :].unsqueeze(1)))
        return self._ramanujan_score_from_r(r)

    def _logits_from_score(self, s: torch.Tensor, j0: int, j1: int, alpha_scale: torch.Tensor | None, temp_scale: torch.Tensor | None, bias_i: torch.Tensor | None) -> torch.Tensor:
        a_scale = 1.0 if alpha_scale is None else alpha_scale
        logits = (self.cfg.alpha * a_scale) * s
        if self.w_mod is not None:
            logits = logits + self.cfg.beta * self.w_mod[:, j0:j1].unsqueeze(0)
        if bias_i is not None:
            logits = logits + bias_i.unsqueeze(-1)
        t_scale = 1.0 if temp_scale is None else temp_scale
        return logits / (max(self.cfg.temp, 1e-6) * t_scale)

    def _build_operator_low_mem(self, z: torch.Tensor, alpha_scale: torch.Tensor | None = None, temp_scale: torch.Tensor | None = None, bias_i: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q, _ = z.shape
        device = z.device
        dtype = z.dtype
        bs = max(1, self.cfg.block_size)

        row_max = torch.full((bsz, q), -float("inf"), dtype=dtype, device=device)
        for j0 in range(0, q, bs):
            j1 = min(q, j0 + bs)
            s_blk = self._score_block(z, j0, j1)
            logits = self._logits_from_score(s_blk, j0, j1, alpha_scale, temp_scale, bias_i)
            row_max = torch.maximum(row_max, logits.max(dim=-1).values)

        denom = torch.zeros((bsz, q), dtype=dtype, device=device)
        for j0 in range(0, q, bs):
            j1 = min(q, j0 + bs)
            s_blk = self._score_block(z, j0, j1)
            logits = self._logits_from_score(s_blk, j0, j1, alpha_scale, temp_scale, bias_i)
            denom = denom + torch.exp(logits - row_max.unsqueeze(-1)).sum(dim=-1)

        a = torch.zeros((bsz, q, q), dtype=dtype, device=device)
        s = torch.zeros((bsz, q, q), dtype=dtype, device=device)
        for j0 in range(0, q, bs):
            j1 = min(q, j0 + bs)
            s_blk = self._score_block(z, j0, j1)
            logits = self._logits_from_score(s_blk, j0, j1, alpha_scale, temp_scale, bias_i)
            probs = torch.exp(logits - row_max.unsqueeze(-1)) / denom.unsqueeze(-1).clamp_min(1e-12)
            a[:, :, j0:j1] = probs
            s[:, :, j0:j1] = s_blk
        return a, s

    def build_operator(self, z: torch.Tensor, alpha_scale: torch.Tensor | None = None, temp_scale: torch.Tensor | None = None, bias_i: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.mode == "low_mem":
            return self._build_operator_low_mem(z, alpha_scale=alpha_scale, temp_scale=temp_scale, bias_i=bias_i)

        r = self._relative_phasor_full(z)
        s = self._ramanujan_score_from_r(r)
        a_scale = 1.0 if alpha_scale is None else alpha_scale
        logits = (self.cfg.alpha * a_scale) * s
        if self.w_mod is not None:
            logits = logits + self.cfg.beta * self.w_mod.unsqueeze(0)
        if bias_i is not None:
            logits = logits + bias_i.unsqueeze(-1)
        t_scale = 1.0 if temp_scale is None else temp_scale
        a = torch.softmax(logits / (max(self.cfg.temp, 1e-6) * t_scale), dim=-1)
        return a, s

    def _relative_summary(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self.cfg.mode != "low_mem":
            r = self._relative_phasor_full(z)
            return (a.unsqueeze(-1) * r).sum(dim=2)
        bsz, q, _ = z.shape
        bs = max(1, self.cfg.block_size)
        m = torch.zeros((bsz, q, 2), device=z.device, dtype=z.dtype)
        zi = z.unsqueeze(2)
        for j0 in range(0, q, bs):
            j1 = min(q, j0 + bs)
            r_blk = _mul(zi, _conj(z[:, j0:j1, :].unsqueeze(1)))
            m = m + (a[:, :, j0:j1].unsqueeze(-1) * r_blk).sum(dim=2)
        return m

    def _router_features(self, z: torch.Tensor, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        _ = s
        m = self._relative_summary(z, a)
        m_re = m[..., 0]
        m_im = m[..., 1]
        m_abs = torch.sqrt((m * m).sum(dim=-1).clamp_min(1e-12))
        entropy = -(a * torch.log(a.clamp_min(1e-12))).sum(dim=-1)
        diag = torch.diagonal(a, dim1=1, dim2=2)
        return torch.stack([m_re, m_im, m_abs, entropy, diag], dim=-1)

    def _routing(self, feats: torch.Tensor, step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(feats)
        tau = max(self.cfg.routing_tau_min, self.cfg.routing_tau * (self.cfg.routing_tau_decay ** step_idx))
        if self.cfg.routing_gumbel:
            pi = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            pi = torch.softmax(logits, dim=-1)
        return pi, logits

    def _map_deltas(self, feats: torch.Tensor) -> torch.Tensor:
        outs = []
        for m in self.maps:
            raw = m(feats)
            outs.append(self.cfg.delta_max * torch.tanh(raw))
        return torch.stack(outs, dim=-1)

    def _phase_noise_sigma(self, step_idx: int) -> float:
        if not self.cfg.noise_enabled:
            return 0.0
        if self.cfg.num_steps <= 1:
            return self.cfg.noise_sigma_min
        t = step_idx / float(self.cfg.num_steps - 1)
        return (1.0 - t) * self.cfg.noise_sigma0 + t * self.cfg.noise_sigma_min

    def _spectral_radius_power(self, a_mean: torch.Tensor) -> torch.Tensor:
        v = torch.randn(a_mean.shape[0], device=a_mean.device, dtype=a_mean.dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(max(1, self.cfg.power_iter_steps)):
            v = a_mean @ v
            v = v / (v.norm() + 1e-12)
        return torch.dot(v, a_mean @ v).abs()

    def _forward_single(self, z0: torch.Tensor, alpha_scale: torch.Tensor | None = None, temp_scale: torch.Tensor | None = None, bias_i: torch.Tensor | None = None, delta_scale: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.cfg.enabled:
            return _renorm(z0), {
                "attn_entropy": [],
                "spectral_radius": [],
                "router_entropy": [],
                "unit_dev_max": [0.0],
                "row_sum_error_max": [0.0],
                "router_entropy_reg": torch.zeros((), device=z0.device, dtype=z0.dtype),
            }
        z = _renorm(z0)
        debug: dict[str, Any] = {
            "attn_entropy": [],
            "spectral_radius": [],
            "router_entropy": [],
            "unit_dev_max": [],
            "row_sum_error_max": [],
        }
        if self.cfg.debug_store_a:
            debug["A_steps"] = []
            debug["S_steps"] = []
        router_ent_accum = z.new_zeros(())

        for s_idx in range(self.cfg.num_steps):
            a, score = self.build_operator(z, alpha_scale=alpha_scale, temp_scale=temp_scale, bias_i=bias_i)
            feats = self._router_features(z, a, score)
            pi, _ = self._routing(feats, s_idx)
            delta_k = self._map_deltas(feats)
            delta = _circular_mean(delta_k, pi)
            if delta_scale is not None:
                delta = delta * delta_scale.view(delta.shape[0], 1)

            z = _renorm(_mul(z, _exp_i(delta)))
            sigma = self._phase_noise_sigma(s_idx)
            if sigma > 0.0:
                eps = torch.randn_like(delta) * sigma
                z = _renorm(_mul(z, _exp_i(eps)))

            if self.cfg.detach_every is not None and self.cfg.detach_every > 0 and (s_idx + 1) % int(self.cfg.detach_every) == 0:
                z = z.detach()

            row_err = (a.sum(dim=-1) - 1.0).abs().max()
            unit_dev = (torch.sqrt((z * z).sum(dim=-1)) - 1.0).abs().max()
            ent = -(a * torch.log(a.clamp_min(1e-12))).sum(dim=-1).mean()
            r_ent = -(pi * torch.log(pi.clamp_min(1e-12))).sum(dim=-1).mean()
            router_ent_accum = router_ent_accum + r_ent
            rho = self._spectral_radius_power(a.mean(dim=0))

            debug["attn_entropy"].append(float(ent.detach().item()))
            debug["spectral_radius"].append(float(rho.detach().item()))
            debug["router_entropy"].append(float(r_ent.detach().item()))
            debug["unit_dev_max"].append(float(unit_dev.detach().item()))
            debug["row_sum_error_max"].append(float(row_err.detach().item()))
            if self.cfg.debug_store_a:
                debug["A_steps"].append(a.detach())
                debug["S_steps"].append(score.detach())
            if self.cfg.exact_eig_debug:
                debug.setdefault("eigvals", []).append(torch.linalg.eigvals(a[0]).detach())

        debug["router_entropy_reg"] = self.cfg.router_entropy_weight * (router_ent_accum / max(1, self.cfg.num_steps))
        return z, debug

    def _slow_controls(self, z_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, t_len, q, _ = z_seq.shape
        p = max(1, int(self.cfg.slow_pool))
        n_blocks = (t_len + p - 1) // p
        feats = z_seq.new_zeros((bsz, n_blocks, 3))
        for b in range(n_blocks):
            t0, t1 = b * p, min(t_len, (b + 1) * p)
            z_win = z_seq[:, t0:t1]
            z_blk = z_win.reshape(bsz * (t1 - t0), q, 2)
            a_blk, s_blk = self.build_operator(z_blk)
            ent = -(a_blk * torch.log(a_blk.clamp_min(1e-12))).sum(dim=-1).mean(dim=-1).reshape(bsz, t1 - t0).mean(dim=1)
            coh = ((a_blk * s_blk).sum(dim=(-1, -2)) / float(q * q)).reshape(bsz, t1 - t0).mean(dim=1)
            if (t1 - t0) > 1:
                dz = _mul(z_win[:, 1:], _conj(z_win[:, :-1]))
                # Winding number approximation: sum(angle(dz)) / 2pi
                ang_dz = torch.atan2(dz[..., 1], dz[..., 0])
                winding = ang_dz.sum(dim=1) / (2.0 * math.pi)
                drift = (1.0 - dz[..., 0]).mean(dim=(1, 2))
                # Add winding density feature
                winding_feat = winding.abs().mean(dim=1)
            else:
                drift = torch.zeros((bsz,), device=z_seq.device, dtype=z_seq.dtype)
                winding_feat = torch.zeros((bsz,), device=z_seq.device, dtype=z_seq.dtype)
                
            feats[:, b, 0] = coh
            feats[:, b, 1] = ent
            feats[:, b, 2] = drift
            # We only have 3 dims allocated in the GRU input currently. 
            # Overwriting drift with winding-weighted drift for now.
            feats[:, b, 2] = drift + 0.1 * winding_feat
            
        h, _ = self.slow_gru(feats)
        ctrl = self.slow_proj(h)
        alpha_b = 0.5 + torch.sigmoid(ctrl[..., 0:1])
        temp_b = 0.5 + torch.sigmoid(ctrl[..., 1:2])
        delta_b = self.cfg.slow_delta_floor + (self.cfg.slow_delta_ceil - self.cfg.slow_delta_floor) * torch.sigmoid(ctrl[..., 2:3])
        alpha_t = alpha_b.repeat_interleave(p, dim=1)[:, :t_len]
        temp_t = temp_b.repeat_interleave(p, dim=1)[:, :t_len]
        delta_t = delta_b.repeat_interleave(p, dim=1)[:, :t_len]
        return alpha_t, temp_t, delta_t

    def forward_sequence(self, z_seq: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.cfg.enabled:
            return _renorm(z_seq), {
                "attn_entropy": [],
                "spectral_radius": [],
                "router_entropy": [],
                "unit_dev_max": [0.0],
                "row_sum_error_max": [0.0],
                "slow_alpha_mean": 1.0,
                "slow_temp_mean": 1.0,
                "slow_delta_mean": 1.0,
            }
        z_seq = _renorm(z_seq)
        bsz, t_len, q, _ = z_seq.shape
        if self.cfg.slow_clock_enabled:
            alpha_t, temp_t, delta_t = self._slow_controls(z_seq)
        else:
            alpha_t = z_seq.new_ones((bsz, t_len, 1))
            temp_t = z_seq.new_ones((bsz, t_len, 1))
            delta_t = z_seq.new_ones((bsz, t_len, 1))

        outs = []
        dbg = {"attn_entropy": [], "spectral_radius": [], "router_entropy": [], "unit_dev_max": [], "row_sum_error_max": []}
        z_prev = z_seq[:, 0]
        for t in range(t_len):
            zt, d = self._forward_single(
                z_seq[:, t],
                alpha_scale=alpha_t[:, t].unsqueeze(-1),
                temp_scale=temp_t[:, t].unsqueeze(-1),
                delta_scale=delta_t[:, t],
            )
            if self.cfg.slow_clock_enabled and t > 0:
                # Gauge-safe temporal smoothing to bound long-horizon drift.
                mix = (1.0 - delta_t[:, t]).view(bsz, 1, 1).clamp(0.05, 0.95)
                zt = _renorm((1.0 - mix) * zt + mix * z_prev)
            z_prev = zt
            outs.append(zt)
            for k in dbg:
                dbg[k].extend(d.get(k, []))
        dbg["slow_alpha_mean"] = float(alpha_t.mean().detach().item())
        dbg["slow_temp_mean"] = float(temp_t.mean().detach().item())
        dbg["slow_delta_mean"] = float(delta_t.mean().detach().item())
        return torch.stack(outs, dim=1), dbg

    def forward(self, z0: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._forward_single(z0)
