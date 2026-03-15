import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
from pathlib import Path
from rafa_clutch_transformer_upgraded import RAFAClutchTransformerUpgraded as RAFAClutchTransformer
from phase_native_ifs import PhaseNativeIFS

def wrap_phase(x):
    tau = 2 * torch.pi
    return (x + torch.pi) % tau - torch.pi

def _resize_solver_control(v: torch.Tensor, target_t: int) -> torch.Tensor:
    if v is None: return None
    if v.size(1) == target_t: return v
    if v.dim() == 2:
        return F.interpolate(v.unsqueeze(1), size=target_t, mode="linear", align_corners=True).squeeze(1)
    elif v.dim() == 3:
        return F.interpolate(v.transpose(1, 2), size=target_t, mode="linear", align_corners=True).transpose(1, 2)
    return v

def _merge_solver_controls(base, injected, target_t):
    if not base and not injected: return None
    merged = {}
    for src in (base or {}, injected or {}):
        for key, value in src.items():
            merged[key] = _resize_solver_control(value, target_t)
    return merged

def _load_hyena_operator():
    here = Path(__file__).resolve().parent
    hyena_py = here / "third_party" / "safari" / "standalone_hyena.py"
    spec = importlib.util.spec_from_file_location("safari_standalone_hyena", str(hyena_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HyenaOperator

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        b, seq_len, d_model = x.size()
        qkv = self.qkv(x).view(b, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scores = torch.matmul(q.transpose(1, 2), k.permute(0, 2, 3, 1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2).reshape(b, seq_len, d_model)
        return self.out_proj(out)

class BlackwellNativeSolver(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.w_ih = nn.Linear(d_model, 3 * d_model)
        self.w_hh = nn.Linear(d_model, 3 * d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
    def forward(self, x_t, h_prev, history):
        ih, hh = self.w_ih(x_t), self.w_hh(h_prev)
        i_r, i_z, i_n = ih.chunk(3, dim=-1)
        h_r, h_z, h_n = hh.chunk(3, dim=-1)
        r, z = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_mid = (1 - z) * n + z * h_prev
        hist_ext = torch.cat([history, h_mid.unsqueeze(1)], dim=1) if history is not None else h_mid.unsqueeze(1)
        h_new = self.attn(hist_ext)[:, -1, :]
        return h_new, hist_ext

class Gear(nn.Module):
    def __init__(self, cfg, gear_cfg, text_dim):
        super().__init__()
        freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
        self.d_model = gear_cfg.get("d_model", 256)
        self.n_heads = gear_cfg.get("n_heads", 4)
        self.n_layers = gear_cfg.get("n_layers", 2)
        self.sequence_backend = "hyena" # FORCED FOR BLACKWELL
        input_dim = (6 * freq_bins) + text_dim
        self.proj = nn.Linear(input_dim, self.d_model)
        HyenaOperator = _load_hyena_operator()
        self.hyena = HyenaOperator(d_model=self.d_model, l_max=512, order=2, filter_order=64)
        self.proj_ph = nn.Linear(self.d_model, freq_bins)
        self.proj_mag = nn.Linear(self.d_model, freq_bins)
    def forward(self, mag, phase, text_emb):
        # Simplified for sm_120 baseline
        x = torch.cat([mag, mag, mag, mag, mag, mag], dim=1).transpose(1, 2)
        x = self.proj(x)
        seq_out = self.hyena(x)
        return self.proj_ph(seq_out).transpose(1, 2), self.proj_mag(seq_out).transpose(1, 2), {}

class RAFA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_conditioned = False
        self.gears = nn.ModuleList([Gear(cfg, gc, 0) for gc in cfg["model"]["gears"]])
        freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
        self.phase_clutch = RAFAClutchTransformer(len(self.gears), freq_bins, 128, 2, 4, [4]*len(self.gears), 0.1, 1e-3, False, "repeat")
        self.mag_clutch = RAFAClutchTransformer(len(self.gears), freq_bins, 128, 2, 4, [4]*len(self.gears), 0.1, 1e-3, False, "repeat")
        self.phase_fuse = nn.Conv2d(len(self.gears), 1, 1)
        self.mag_fuse = nn.Conv2d(len(self.gears), 1, 1)
        self.phase_solver = PhaseNativeIFS(freq_bins, cfg.get("phase_native_ifs", {}))
        self.learned_freqs = nn.Parameter(torch.linspace(0.0, torch.pi, freq_bins))
    def forward(self, mag, phase, tokens=None, ifs_text_controls=None, semantic_controls=None):
        preds_p, preds_m = [], []
        for gear in self.gears:
            p, m, _ = gear(mag, phase, None)
            preds_p.append(p); preds_m.append(m)
        f_p, _, _ = self.phase_clutch(torch.stack(preds_p, dim=1).permute(0, 1, 3, 2))
        f_m, _, _ = self.mag_clutch(torch.stack(preds_m, dim=1).permute(0, 1, 3, 2))
        p_seed = self.phase_fuse(f_p).squeeze(1).transpose(1, 2)
        final_mag = torch.abs(self.mag_fuse(f_m).squeeze(1).transpose(1, 2))
        z_in = torch.stack([torch.cos(p_seed), torch.sin(p_seed)], dim=-1).permute(0, 2, 1, 3)
        z_out, ifs_debug = self.phase_solver.forward_sequence(z_in, tension=semantic_controls.get("semantic_tension") if semantic_controls else None)
        zf = z_out.permute(0, 2, 1, 3)
        final_p = torch.atan2(zf[..., 1], zf[..., 0] + 1e-12)
        return final_p, final_mag, {"phase_state": zf, "ifs_debug": ifs_debug}, {}
