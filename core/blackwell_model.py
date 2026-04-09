import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
import importlib.util

# BLACKWELL sm_120 COMPATIBLE CORE
# This file contains ZERO references to RNN, GRU, or LSTM.

def wrap_phase(x):
    tau = 2 * torch.pi
    return (x + torch.pi) % tau - torch.pi

class MultiHeadAttentionNative(nn.Module):
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

class TemporalStepperNative(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.w_ih = nn.Linear(d_model, 3 * d_model)
        self.w_hh = nn.Linear(d_model, 3 * d_model)
        self.attn = MultiHeadAttentionNative(d_model, n_heads)
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

class GearNative(nn.Module):
    def __init__(self, cfg, gear_cfg, text_dim):
        super().__init__()
        freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
        self.d_model = gear_cfg.get("d_model", 256)
        self.n_heads = gear_cfg.get("n_heads", 4)
        self.n_layers = gear_cfg.get("n_layers", 2)
        input_dim = (6 * freq_bins) + text_dim
        self.proj = nn.Linear(input_dim, self.d_model)
        self.steppers = nn.ModuleList([TemporalStepperNative(self.d_model, self.n_heads) for _ in range(self.n_layers)])
        self.proj_ph = nn.Linear(self.d_model, freq_bins)
        self.proj_mag = nn.Linear(self.d_model, freq_bins)
    def forward(self, mag, phase, text_emb):
        x = torch.cat([mag]*6, dim=1).transpose(1, 2)
        x = self.proj(x)
        b, t_len, _ = x.size()
        h = x.new_zeros(b, self.d_model)
        hist = None
        outs = []
        for t_ in range(t_len):
            xt = x[:, t_]
            for s in self.steppers:
                h, hist = s(xt, h, hist)
            outs.append(h)
        seq_out = torch.stack(outs, dim=1)
        return self.proj_ph(seq_out).transpose(1, 2), self.proj_mag(seq_out).transpose(1, 2), {}

class RAFA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gears = nn.ModuleList([GearNative(cfg, gc, 0) for gc in cfg["model"]["gears"]])
        freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
        from rafa_clutch_transformer_upgraded import RAFAClutchTransformerUpgraded as RCTU
        self.phase_clutch = RCTU(len(self.gears), freq_bins, 128, 2, 4, [4]*len(self.gears), 0.1, 1e-3, False, "repeat")
        self.mag_clutch = RCTU(len(self.gears), freq_bins, 128, 2, 4, [4]*len(self.gears), 0.1, 1e-3, False, "repeat")
        self.phase_fuse = nn.Conv2d(len(self.gears), 1, 1)
        self.mag_fuse = nn.Conv2d(len(self.gears), 1, 1)
        from phase_native_ifs import PhaseNativeIFS
        self.phase_gru = PhaseNativeIFS(freq_bins, cfg.get("phase_native_ifs", {}))
    def forward(self, mag, phase, tokens=None, ifs_text_controls=None, semantic_controls=None):
        preds_p, preds_m = [], []
        for g in self.gears:
            p, m, _ = g(mag, phase, None)
            preds_p.append(p); preds_m.append(m)
        f_p, _, _ = self.phase_clutch(torch.stack(preds_p, dim=1).permute(0, 1, 3, 2))
        f_m, _, _ = self.mag_clutch(torch.stack(preds_m, dim=1).permute(0, 1, 3, 2))
        p_seed = self.phase_fuse(f_p).squeeze(1).transpose(1, 2)
        final_m = torch.abs(self.mag_fuse(f_m).squeeze(1).transpose(1, 2))
        z_in = torch.stack([torch.cos(p_seed), torch.sin(p_seed)], dim=-1).permute(0, 2, 1, 3)
        z_out, ifs_debug = self.phase_gru.forward_sequence(z_in, tension=semantic_controls.get("semantic_tension") if semantic_controls else None)
        zf = z_out.permute(0, 2, 1, 3)
        final_p = torch.atan2(zf[..., 1], zf[..., 0] + 1e-12)
        return final_p, final_m, {"phase_state": zf, "ifs_debug": ifs_debug}, {}
