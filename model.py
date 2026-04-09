import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
from pathlib import Path
from rafa_clutch_transformer_upgraded import RAFAClutchTransformerUpgraded as RAFAClutchTransformer
# BLACKWELL FIX: No transformers imports allowed
# from transformers import AutoModel
from phase_native_ifs import PhaseNativeIFS
from hf_local import resolve_hf_pretrained_path

def wrap_phase(x):
    tau = 2 * torch.pi
    return (x + torch.pi) % tau - torch.pi


def _resize_solver_control(v: torch.Tensor, target_t: int) -> torch.Tensor:
    if v is None:
        return None
    # v: [B, T] or [B, T, C]
    if v.size(1) == target_t:
        return v
    
    if v.dim() == 2:
        # [B, T] -> [B, 1, T] -> [B, 1, T_new] -> [B, T_new]
        return F.interpolate(v.unsqueeze(1), size=target_t, mode="linear", align_corners=True).squeeze(1)
    elif v.dim() == 3:
        # [B, T, C] -> [B, C, T] -> [B, C, T_new] -> [B, T_new, C]
        return F.interpolate(v.transpose(1, 2), size=target_t, mode="linear", align_corners=True).transpose(1, 2)
    return v


def _merge_solver_controls(
    base_controls: dict[str, torch.Tensor] | None,
    injected_controls: dict[str, torch.Tensor] | None,
    target_t: int,
) -> dict[str, torch.Tensor] | None:
    if not base_controls and not injected_controls:
        return None
    merged: dict[str, torch.Tensor] = {}
    for src in (base_controls or {}, injected_controls or {}):
        for key, value in src.items():
            if key == "semantic_tension":
                merged[key] = _resize_solver_control(value, target_t)
                continue
            merged[key] = _resize_solver_control(value, target_t)
    if base_controls and injected_controls:
        for key, inj in injected_controls.items():
            if key not in base_controls:
                continue
            if key == "semantic_tension":
                continue
            base = _resize_solver_control(base_controls[key], target_t)
            inj = _resize_solver_control(inj, target_t)
            if key == "qset_scale":
                mix = base * inj
                qn = max(1, mix.size(-1))
                merged[key] = mix / mix.mean(dim=-1, keepdim=True).clamp_min(1e-6)
                merged[key] = merged[key].clamp_min(1e-4)
                merged[key] = merged[key] * (float(qn) / merged[key].mean(dim=-1, keepdim=True).clamp_min(1e-6))
            else:
                merged[key] = base * inj
    return merged

def apply_rotary_pos_emb(q, k):
    b, h, seq_len, d = q.size()
    half = d // 2
    pos = torch.arange(seq_len, dtype=q.dtype, device=q.device)
    freq = torch.arange(half, dtype=q.dtype, device=q.device) / float(half)
    angles = pos[:, None] * (10000.0 ** (-freq[None, :]))
    sin = torch.sin(angles)[None, None, :, :]
    cos = torch.cos(angles)[None, None, :, :]
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


def _load_hyena_operator():
    here = Path(__file__).resolve().parent
    hyena_py = here / "third_party" / "safari" / "standalone_hyena.py"
    if not hyena_py.exists():
        raise RuntimeError(f"Hyena source not found at {hyena_py}")
    spec = importlib.util.spec_from_file_location("safari_standalone_hyena", str(hyena_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Hyena module from {hyena_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "HyenaOperator"):
        raise RuntimeError("HyenaOperator not found in standalone_hyena.py")
    return mod.HyenaOperator

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        qkv = self.qkv(x).view(b, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q, k = apply_rotary_pos_emb(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, d_model)
        return self.out_proj(out)

class BlackwellNativeSolver(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # BLACKWELL FIX: Use native matmul-based cell, no kernel strings allowed
        self.d_model = d_model
        self.w_ih = nn.Linear(d_model, 3 * d_model)
        self.w_hh = nn.Linear(d_model, 3 * d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)

    def forward(self, x_t, h_prev, history):
        ih = self.w_ih(x_t)
        hh = self.w_hh(h_prev)
        i_r, i_z, i_n = ih.chunk(3, dim=-1)
        h_r, h_z, h_n = hh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_mid = (1 - z) * n + z * h_prev
        
        if history is None or history.size(1) == 0:
            hist_ext = h_mid.unsqueeze(1)
        else:
            hist_ext = torch.cat([history, h_mid.unsqueeze(1)], dim=1)
        attn_out = self.attn(hist_ext)
        h_new = attn_out[:, -1, :]
        return h_new, hist_ext

class Gear(nn.Module):
    def __init__(self, cfg, gear_cfg, text_dim):
        super().__init__()
        n_fft = cfg["data"]["stft"]["n_fft"]
        freq_bins = n_fft // 2 + 1
        self.include_mag = gear_cfg.get("include_mag", True)
        self.include_phase = gear_cfg.get("include_phase", True)
        self.ms = gear_cfg.get("time_scale", 1)
        self.d_model = gear_cfg.get("d_model", 256)
        self.n_heads = gear_cfg.get("n_heads", 4)
        self.n_layers = gear_cfg.get("n_layers", 2)
        self.sequence_backend = str(gear_cfg.get("sequence_backend", cfg["model"].get("sequence_backend", "attn_solver")))
        self.hyena_qset = [int(q) for q in gear_cfg.get("hyena_qset", cfg["model"].get("hyena_qset", [2, 3, 4, 5, 6, 8, 12]))]
        self.hyena_gate_scale = float(gear_cfg.get("hyena_gate_scale", cfg["model"].get("hyena_gate_scale", 0.25)))
        self.hyena_conductor_enabled = bool(gear_cfg.get("hyena_conductor_enabled", cfg["model"].get("hyena_conductor_enabled", False)))
        self.mag_s = nn.Parameter(torch.tensor(1.0))
        if self.sequence_backend == "hyena":
            input_dim = ((6 if self.include_mag else 5) * freq_bins) + text_dim
        else:
            input_dim = ((freq_bins if self.include_mag else 0) +
                         (freq_bins if self.include_phase else 0) +
                         3 * freq_bins + text_dim)
        self.proj = nn.Linear(input_dim, self.d_model)
        self.temporal_solvers = None
        self.hyena = None
        self.hyena_gate_mlp = None
        if self.sequence_backend == "attn_solver":
            self.temporal_solvers = nn.ModuleList([
                BlackwellNativeSolver(self.d_model, self.n_heads)
                for _ in range(self.n_layers)
            ])
        elif self.sequence_backend == "hyena":
            HyenaOperator = _load_hyena_operator()
            hyena_lmax = int(gear_cfg.get("hyena_l_max", cfg["model"].get("hyena_l_max", 512)))
            hyena_order = int(gear_cfg.get("hyena_order", cfg["model"].get("hyena_order", 2)))
            hyena_filter_order = int(gear_cfg.get("hyena_filter_order", cfg["model"].get("hyena_filter_order", 64)))
            hyena_dropout = float(gear_cfg.get("hyena_dropout", cfg["model"].get("hyena_dropout", 0.0)))
            self.hyena = HyenaOperator(
                d_model=self.d_model,
                l_max=hyena_lmax,
                order=hyena_order,
                filter_order=hyena_filter_order,
                dropout=hyena_dropout,
            )
            qfeat = max(1, len(self.hyena_qset))
            hdim = max(16, min(64, self.d_model // 2))
            self.hyena_gate_mlp = nn.Sequential(
                nn.Linear(qfeat, hdim),
                nn.SiLU(),
                nn.Linear(hdim, qfeat),
            )
            if self.hyena_conductor_enabled:
                # alpha/temp/delta + qset-scale controls.
                self.hyena_ctrl_head = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, 3 + qfeat),
                )
            else:
                self.hyena_ctrl_head = None
        else:
            raise ValueError(f"Unsupported sequence_backend: {self.sequence_backend}")
        self.proj_ph = nn.Linear(self.d_model, freq_bins)
        self.proj_mag = nn.Linear(self.d_model, freq_bins)

    def compute_kinematics(self, mag, phase):
        dphi = wrap_phase(phase[..., 1:] - phase[..., :-1])
        dmag = mag[..., 1:] - mag[..., :-1]
        arc_length = torch.sqrt(dphi ** 2 + dmag ** 2)
        velocity = arc_length
        acceleration = velocity[..., 1:] - velocity[..., :-1]
        pad = lambda x: torch.cat([torch.zeros_like(x[..., :1]), x], dim=-1)
        velocity = pad(velocity)
        acceleration = pad(pad(acceleration))
        arc_length = pad(arc_length)
        return velocity, acceleration, arc_length

    def _relative_phase(self, phase):
        dphi = wrap_phase(phase[..., 1:] - phase[..., :-1])
        dphi = torch.cat([torch.zeros_like(phase[..., :1]), dphi], dim=-1)
        return dphi

    def _hyena_invariant_features(self, mag_s, phase):
        vel, acc, arc = self.compute_kinematics(mag_s, phase)
        dphi = self._relative_phase(phase)
        cos_dphi = torch.cos(dphi)
        sin_dphi = torch.sin(dphi)
        feats = []
        if self.include_mag:
            feats.append(mag_s)
        feats.extend([vel, acc, arc, cos_dphi, sin_dphi])
        min_len = min(f.shape[-1] for f in feats)
        feats = [f[..., :min_len] for f in feats]
        return feats, min_len

    def _hyena_ramanujan_gate(self, phase, t_len):
        # Build continuous major-arc preferences from relative phase only.
        bsz, _, _ = phase.shape
        scores = []
        for q in self.hyena_qset:
            q = int(q)
            if q <= 0 or q >= t_len:
                scores.append(torch.zeros((bsz, t_len), device=phase.device, dtype=phase.dtype))
                continue
            d = wrap_phase(phase[..., q:t_len] - phase[..., : t_len - q])
            c = torch.cos(d).mean(dim=1)  # [B, T-q], global phase-invariant
            pad = torch.zeros((bsz, q), device=phase.device, dtype=phase.dtype)
            scores.append(torch.cat([pad, c], dim=-1))
        if not scores:
            return torch.ones((bsz, t_len, 1), device=phase.device, dtype=phase.dtype)
        s = torch.stack(scores, dim=-1)  # [B,T,|Q|]
        logits = self.hyena_gate_mlp(s)
        pi = torch.softmax(logits, dim=-1)
        mix = (pi * s).sum(dim=-1, keepdim=True)
        return 1.0 + self.hyena_gate_scale * torch.tanh(mix)

    def forward(self, mag, phase, text_emb):
        mag_s = mag * self.mag_s
        if self.sequence_backend == "hyena":
            feats, min_len = self._hyena_invariant_features(mag_s, phase)
        else:
            vel, acc, arc = self.compute_kinematics(mag_s, phase)
            feats = []
            if self.include_mag:
                feats.append(mag_s)
            if self.include_phase:
                feats.append(phase)
            feats.extend([vel, acc, arc])
            min_len = min(f.shape[-1] for f in feats)
            feats = [f[..., :min_len] for f in feats]
        x = torch.cat(feats, dim=1).transpose(1, 2)
        if text_emb is not None and text_emb.numel() > 0:
            t = text_emb.float()
            if t.dim() == 2:
                t = t.unsqueeze(1)
            t = t.expand(-1, x.size(1), -1)
            x = torch.cat([x, t], dim=2)
        elif self.proj.in_features > x.shape[-1]:
            padding_size = self.proj.in_features - x.shape[-1]
            padding = torch.zeros(x.shape[0], x.shape[1], padding_size, device=x.device)
            x = torch.cat([x, padding], dim=2)

        x = self.proj(x)
        hyena_controls = None
        if self.sequence_backend == "hyena":
            seq_out = self.hyena(x)
            gate = self._hyena_ramanujan_gate(phase[..., :x.size(1)], x.size(1))
            seq_out = seq_out * gate
            if self.hyena_ctrl_head is not None:
                ctrl = self.hyena_ctrl_head(seq_out)  # [B,T,3+|qset|]
                qn = max(1, len(self.hyena_qset))
                alpha_raw = ctrl[..., 0:1]
                temp_raw = ctrl[..., 1:2]
                delta_raw = ctrl[..., 2:3]
                qset_raw = ctrl[..., 3:3 + qn]
                hyena_controls = {
                    "alpha_scale": 0.5 + torch.sigmoid(alpha_raw),  # [0.5, 1.5]
                    "temp_scale": 0.5 + torch.sigmoid(temp_raw),    # [0.5, 1.5]
                    "delta_scale": 0.5 + torch.sigmoid(delta_raw),  # [0.5, 1.5]
                    "qset_scale": torch.softmax(qset_raw, dim=-1) * float(qn),  # mean ~1
                }
        else:
            b, t_len, _ = x.size()
            h = x.new_zeros(b, self.d_model)
            history = None
            outs = []
            for t_ in range(t_len):
                x_t = x[:, t_, :]
                for solver in self.temporal_solvers:
                    h, history = solver(x_t, h, history)
                outs.append(h)
            seq_out = torch.stack(outs, dim=1)
        if self.ms > 1:
            seq_out = F.interpolate(
                seq_out.transpose(1, 2),
                scale_factor=self.ms,
                mode='linear',
                align_corners=True
            ).transpose(1, 2)
            seq_out = seq_out[:, :phase.shape[-1], :]
        pd = self.proj_ph(seq_out).transpose(1, 2)
        md = self.proj_mag(seq_out).transpose(1, 2)
        
        # Ensure F dimension matches input exactly
        target_f = mag.size(1)
        if pd.size(1) != target_f:
            pd = F.interpolate(pd.unsqueeze(1), size=(target_f, pd.size(2)), mode='bilinear', align_corners=True).squeeze(1)
            md = F.interpolate(md.unsqueeze(1), size=(target_f, md.size(2)), mode='bilinear', align_corners=True).squeeze(1)
            
        extras = {"hyena_controls": hyena_controls} if hyena_controls is not None else {}
        return pd, md, extras

class ContinuousGaborLayer(nn.Module):
    """
    Learned Gabor Transform Layer (RAFA-PC v3.0).
    Uses the learned frequency grid to project the raw waveform into the lattice.
    Replaces torch.stft when enabled.
    """
    def __init__(self, n_fft, hop_length, win_length, sample_rate, learned_freqs):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.sr = sample_rate
        self.freqs = learned_freqs # Parameter from RAFA model
        # Learnable bandwidths (sigma) for the Gabor filters
        self.bandwidths = nn.Parameter(torch.ones_like(learned_freqs) * 10.0) 

    def forward(self, audio):
        # audio: [B, T_samples]
        B, T = audio.shape
        # Create time grid for the windows
        n_frames = 1 + (T - self.win) // self.hop
        
        # This is computationally heavy for a full unroll, so we use a conv approximation
        # Construct the filterbank from the current freqs and bandwidths
        t = torch.arange(-(self.win//2), (self.win//2), device=audio.device).float() / self.sr
        
        filters_re = []
        filters_im = []
        
        # We only use a subset or the full grid? Full grid.
        # w_k = exp(-0.5 * (t/sigma)^2) * exp(i * omega * t)
        for i, omega in enumerate(self.freqs):
            sigma = 1.0 / (self.bandwidths[i].abs() + 1e-5)
            env = torch.exp(-0.5 * (t * sigma)**2)
            osc_re = torch.cos(omega * t * self.sr) # omega is normalized [0, pi] usually
            osc_im = torch.sin(omega * t * self.sr)
            filters_re.append(env * osc_re)
            filters_im.append(env * osc_im)
            
        filters_re = torch.stack(filters_re, dim=0).unsqueeze(1) # [F, 1, K]
        filters_im = torch.stack(filters_im, dim=0).unsqueeze(1)
        
        # Convolve
        spec_re = F.conv1d(audio.unsqueeze(1), filters_re, stride=self.hop)
        spec_im = F.conv1d(audio.unsqueeze(1), filters_im, stride=self.hop)
        
        mag = torch.sqrt(spec_re**2 + spec_im**2 + 1e-8)
        phase = torch.atan2(spec_im, spec_re)
        
        return mag, phase


class RAFA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_conditioned = cfg["model"].get("text_conditioned", False)
        if self.text_conditioned:
            text_encoder_id = resolve_hf_pretrained_path(
                cfg["model"]["text_encoder"],
                need_model=True,
            )
            self.text_encoder = AutoModel.from_pretrained(
                text_encoder_id,
                local_files_only=os.path.isdir(text_encoder_id),
            )
            text_dim = int(
                getattr(
                    self.text_encoder.config,
                    "projection_dim",
                    getattr(self.text_encoder.config, "hidden_size", 768),
                )
            )
        else:
            self.text_encoder = None
            text_dim = 0

        gear_cfgs = cfg["model"]["gears"]
        self.gears = nn.ModuleList([
            Gear(cfg, gc, text_dim) for gc in gear_cfgs
        ])
        n_gears = len(self.gears)
        freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
        clutch_tau = float(cfg.get("model", {}).get("clutch_gate_tau", 1.0))
        clutch_flux_w = float(cfg.get("model", {}).get("clutch_flux_weight", 0.0))
        clutch_peak_w = float(cfg.get("model", {}).get("clutch_peak_weight", 0.0))

        # Clutch transformers
        self.phase_clutch = RAFAClutchTransformer(
            num_gears=n_gears,
            gear_dim=freq_bins,
            transformer_dim=128,
            num_layers=2,
            num_heads=4,
            pool_kernel=[4] * n_gears,
            dropout=0.1,
            gate_reg_weight=1e-3,
            nonlinear_fusion=False,
            upsample_mode="repeat",
            gate_tau=clutch_tau,
            flux_weight=clutch_flux_w,
            peak_weight=clutch_peak_w,
        )
        self.mag_clutch = RAFAClutchTransformer(
            num_gears=n_gears,
            gear_dim=freq_bins,
            transformer_dim=128,
            num_layers=2,
            num_heads=4,
            pool_kernel=[4] * n_gears,
            dropout=0.1,
            gate_reg_weight=1e-3,
            nonlinear_fusion=False,
            upsample_mode="repeat",
            gate_tau=clutch_tau,
            flux_weight=clutch_flux_w,
            peak_weight=clutch_peak_w,
        )
        self.phase_fuse = nn.Conv2d(n_gears, 1, 1)
        self.mag_fuse = nn.Conv2d(n_gears, 1, 1)

        ifs_cfg = cfg.get("phase_native_ifs", {})
        self.phase_solver = PhaseNativeIFS(freq_bins, ifs_cfg)

        # RAFA‑PC v3.0: learned frequency grid (ablatable via config)
        learn_grid = cfg["model"].get("learn_grid", False)
        if learn_grid:
            # stable init: match exact STFT bin positions
            init_freqs = torch.linspace(0.0, torch.pi, freq_bins)
            self.learned_freqs = nn.Parameter(init_freqs.clone())
        else:
            self.learned_freqs = None

    def forward(self, mag, phase, tokens=None, ifs_text_controls=None, semantic_controls=None):
        if self.text_conditioned and tokens is not None:
            text_out = self.text_encoder(
                input_ids=tokens.get("input_ids"),
                attention_mask=tokens.get("attention_mask")
            )
            if hasattr(text_out, "text_embeds") and text_out.text_embeds is not None:
                text_emb = text_out.text_embeds
            elif hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
                text_emb = text_out.pooler_output
            else:
                text_emb = text_out.last_hidden_state.mean(dim=1)
        else:
            text_emb = mag.new_zeros(mag.size(0), 0)

        phase_preds, mag_preds = [], []
        hyena_ctrls = []
        for gear in self.gears:
            pd, md, g_ext = gear(mag, phase, text_emb)
            phase_preds.append(pd)
            mag_preds.append(md)
            if isinstance(g_ext, dict) and g_ext.get("hyena_controls", None) is not None:
                hyena_ctrls.append(g_ext["hyena_controls"])

        phase_stack = torch.stack(phase_preds, dim=1)
        mag_stack = torch.stack(mag_preds, dim=1)
        
        target_f = mag.size(1)
        target_t = mag.size(2)

        print(f"DEBUG: phase_stack shape: {phase_stack.shape}")
        # phase_stack: [B, C, F_gear, T]
        fused_phase, phase_gates, phase_extras = self.phase_clutch(
            phase_stack.permute(0, 1, 3, 2) # [B, C, T, F_gear]
        )
        # fused_phase: [B, C, T, F_clutch]
        if fused_phase.size(-1) != target_f or fused_phase.size(2) != target_t:
            # Interpolate [B, C, T, F] -> [B, C, target_t, target_f]
            fused_phase = F.interpolate(fused_phase, size=(target_t, target_f), mode='bilinear', align_corners=True)
            
        print(f"DEBUG: fused_phase shape: {fused_phase.shape}")
        fused_mag, mag_gates, mag_extras = self.mag_clutch(
            mag_stack.permute(0, 1, 3, 2)
        )
        if fused_mag.size(-1) != target_f or fused_mag.size(2) != target_t:
            fused_mag = F.interpolate(fused_mag, size=(target_t, target_f), mode='bilinear', align_corners=True)
            
        # [B, C, T, F] -> [B, C, target_t, target_f] for Conv2d fuser expectation [B, C, F, T]
        phase_seed = self.phase_fuse(fused_phase.permute(0, 1, 3, 2)).squeeze(1).transpose(1, 2)
        # self.phase_fuse( [B, C, target_f, target_t] ) -> [B, 1, target_f, target_t] 
        # squeeze -> [B, target_f, target_t], transpose(1,2) -> [B, target_t, target_f]
        # WAIT, sample_diffusion expects [B, F, T]
        
        phase_seed = self.phase_fuse(fused_phase.permute(0, 1, 3, 2)).squeeze(1) # [B, F, T]
        final_mag = torch.abs(self.mag_fuse(fused_mag.permute(0, 1, 3, 2)).squeeze(1)) # [B, F, T]
        
        print(f"DEBUG: phase_seed shape: {phase_seed.shape}")
        print(f"DEBUG: final_mag shape: {final_mag.shape}")

        z0 = torch.stack([torch.cos(phase_seed), torch.sin(phase_seed)], dim=-1)  # [B,F,T,2]
        z_in = z0.permute(0, 2, 1, 3)  # [B,T,F,2]
        t_stride = max(1, int(getattr(self.phase_solver.cfg, "time_stride", 1)))
        hyena_controls = None
        if hyena_ctrls:
            # Aggregate controls across gears and match solver timeline length.
            keys = ("alpha_scale", "temp_scale", "delta_scale", "qset_scale")
            hyena_controls = {}
            for k in keys:
                vals = [c[k] for c in hyena_ctrls if k in c]
                if not vals:
                    continue
                v = torch.stack(vals, dim=0).mean(dim=0)  # [B,T,*]
                hyena_controls[k] = _resize_solver_control(v, z_in.size(1))
        conditioning_controls = _merge_solver_controls(ifs_text_controls, semantic_controls, z_in.size(1))
        ext_controls = _merge_solver_controls(hyena_controls, conditioning_controls, z_in.size(1))
        
        # Explicit tension for impedance head
        semantic_tension = None
        if isinstance(semantic_controls, dict):
            semantic_tension = _resize_solver_control(semantic_controls.get("semantic_tension"), z_in.size(1))

        if t_stride > 1 and z_in.size(1) > 1:
            z_ds = z_in[:, ::t_stride, :, :]
            ext_ds = None
            tension_ds = None
            if isinstance(ext_controls, dict):
                ext_ds = {k: v[:, ::t_stride, ...] for k, v in ext_controls.items()}
            if semantic_tension is not None:
                tension_ds = semantic_tension[:, ::t_stride, ...]
            z_out_ds, ifs_debug = self.phase_solver.forward_sequence(z_ds, ext_controls=ext_ds, tension=tension_ds)
            # safe_atan2
            ds_y, ds_x = z_out_ds[..., 1], z_out_ds[..., 0]
            phi_ds = torch.atan2(ds_y, ds_x + ((ds_y.abs() < 1e-12) & (ds_x.abs() < 1e-12)).float() * 1e-12)
            phi_up = F.interpolate(phi_ds.permute(0, 2, 1), size=z_in.size(1), mode="linear", align_corners=True).permute(0, 2, 1)
            zf = torch.stack([torch.cos(phi_up), torch.sin(phi_up)], dim=-1).permute(0, 2, 1, 3)
        else:
            z_out, ifs_debug = self.phase_solver.forward_sequence(z_in, ext_controls=ext_controls, tension=semantic_tension)
            zf = z_out.permute(0, 2, 1, 3)
        # safe_atan2
        fy, fx = zf[..., 1], zf[..., 0]
        final_phase = torch.atan2(fy, fx + ((fy.abs() < 1e-12) & (fx.abs() < 1e-12)).float() * 1e-12)

        # Layer 5.1: Pressure Valve (Singular Point Safety)
        # If the solver reached high 'heat' (unit deviation), damp the magnitude
        dev = ifs_debug.get("unit_dev_max", [0.0])
        if isinstance(dev, list) and len(dev) > 0:
            max_dev = max(dev)
            if max_dev > 0.1: # Threshold for damping
                # Log-domain damping: reduce magnitude proportional to deviation
                damping = torch.log1p(torch.tensor(max_dev, device=final_mag.device))
                final_mag = final_mag - 0.5 * damping
                
        if isinstance(phase_extras, dict):
            phase_extras = dict(phase_extras)
            phase_extras["ifs_debug"] = ifs_debug
            phase_extras["phase_state"] = zf
            phase_extras["phase_seed_state"] = z0
            phase_extras["phase_gates"] = phase_gates
            phase_extras["learned_freqs"] = self.learned_freqs
            if hyena_controls is not None:
                phase_extras["hyena_controls"] = hyena_controls
            if ifs_text_controls is not None:
                phase_extras["ifs_text_controls"] = {
                    k: _resize_solver_control(v, z_in.size(1)) for k, v in ifs_text_controls.items()
                }
            if semantic_controls is not None:
                phase_extras["semantic_controls"] = {
                    k: _resize_solver_control(v, z_in.size(1)) for k, v in semantic_controls.items()
                }
            if ext_controls is not None:
                phase_extras["solver_controls"] = ext_controls
        else:
            phase_extras = {
                "ifs_debug": ifs_debug,
                "phase_state": zf,
                "phase_seed_state": z0,
                "phase_gates": phase_gates,
                "learned_freqs": self.learned_freqs,
            }
            if hyena_controls is not None:
                phase_extras["hyena_controls"] = hyena_controls
            if ifs_text_controls is not None:
                phase_extras["ifs_text_controls"] = {
                    k: _resize_solver_control(v, z_in.size(1)) for k, v in ifs_text_controls.items()
                }
            if semantic_controls is not None:
                phase_extras["semantic_controls"] = {
                    k: _resize_solver_control(v, z_in.size(1)) for k, v in semantic_controls.items()
                }
            if ext_controls is not None:
                phase_extras["solver_controls"] = ext_controls

        return final_phase, final_mag, phase_extras, mag_extras
