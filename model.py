import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
from pathlib import Path
from rafa_clutch_transformer_upgraded import RAFAClutchTransformerUpgraded as RAFAClutchTransformer
from transformers import AutoModel
from phase_native_ifs import PhaseNativeIFS

def wrap_phase(x):
    tau = 2 * torch.pi
    return (x + torch.pi) % tau - torch.pi

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

class AttnGRUCell(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.gru_cell = nn.GRUCell(d_model, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)

    def forward(self, x_t, h_prev, history):
        h_mid = self.gru_cell(x_t, h_prev)
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
        self.sequence_backend = str(gear_cfg.get("sequence_backend", cfg["model"].get("sequence_backend", "attn_gru")))
        self.hyena_qset = [int(q) for q in gear_cfg.get("hyena_qset", cfg["model"].get("hyena_qset", [2, 3, 4, 5, 6, 8, 12]))]
        self.hyena_gate_scale = float(gear_cfg.get("hyena_gate_scale", cfg["model"].get("hyena_gate_scale", 0.25)))
        self.mag_s = nn.Parameter(torch.tensor(1.0))
        if self.sequence_backend == "hyena":
            # Gauge-invariant features only: magnitude + relative-phase observables.
            # [mag?] + [vel, acc, arc, cos(dphi), sin(dphi)] => (5 + mag_flag) * F
            input_dim = ((6 if self.include_mag else 5) * freq_bins) + text_dim
        else:
            input_dim = ((freq_bins if self.include_mag else 0) +
                         (freq_bins if self.include_phase else 0) +
                         3 * freq_bins + text_dim)
        self.proj = nn.Linear(input_dim, self.d_model)
        self.grus = None
        self.hyena = None
        self.hyena_gate_mlp = None
        if self.sequence_backend == "attn_gru":
            self.grus = nn.ModuleList([
                AttnGRUCell(self.d_model, self.n_heads)
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
        if self.sequence_backend == "hyena":
            seq_out = self.hyena(x)
            gate = self._hyena_ramanujan_gate(phase[..., :x.size(1)], x.size(1))
            seq_out = seq_out * gate
        else:
            b, t_len, _ = x.size()
            h = x.new_zeros(b, self.d_model)
            history = None
            outs = []
            for t_ in range(t_len):
                x_t = x[:, t_, :]
                for gru in self.grus:
                    h, history = gru(x_t, h, history)
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
        return pd, md

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
            self.text_encoder = AutoModel.from_pretrained(
                cfg["model"]["text_encoder"]
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
            # initialise to evenly spaced values between 0 and π
            init_freqs = torch.linspace(0.0, torch.pi, freq_bins)
            self.learned_freqs = nn.Parameter(init_freqs.clone())
        else:
            self.learned_freqs = None

    def forward(self, mag, phase, tokens=None):
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
        for gear in self.gears:
            pd, md = gear(mag, phase, text_emb)
            phase_preds.append(pd)
            mag_preds.append(md)

        phase_stack = torch.stack(phase_preds, dim=1)
        mag_stack = torch.stack(mag_preds, dim=1)

        fused_phase, phase_gates, phase_extras = self.phase_clutch(
            phase_stack.permute(0, 1, 3, 2)
        )
        fused_mag, mag_gates, mag_extras = self.mag_clutch(
            mag_stack.permute(0, 1, 3, 2)
        )

        phase_seed = self.phase_fuse(fused_phase).squeeze(1).transpose(1, 2)
        final_mag = self.mag_fuse(fused_mag).squeeze(1).transpose(1, 2)

        z0 = torch.stack([torch.cos(phase_seed), torch.sin(phase_seed)], dim=-1)  # [B,F,T,2]
        z_in = z0.permute(0, 2, 1, 3)  # [B,T,F,2]
        t_stride = max(1, int(getattr(self.phase_solver.cfg, "time_stride", 1)))
        if t_stride > 1 and z_in.size(1) > 1:
            z_ds = z_in[:, ::t_stride, :, :]
            z_out_ds, ifs_debug = self.phase_solver.forward_sequence(z_ds)
            phi_ds = torch.atan2(z_out_ds[..., 1], z_out_ds[..., 0])  # [B,Td,F]
            phi_up = F.interpolate(phi_ds.permute(0, 2, 1), size=z_in.size(1), mode="linear", align_corners=True).permute(0, 2, 1)
            zf = torch.stack([torch.cos(phi_up), torch.sin(phi_up)], dim=-1).permute(0, 2, 1, 3)
        else:
            z_out, ifs_debug = self.phase_solver.forward_sequence(z_in)
            zf = z_out.permute(0, 2, 1, 3)
        final_phase = torch.atan2(zf[..., 1], zf[..., 0])

        if isinstance(phase_extras, dict):
            phase_extras = dict(phase_extras)
            phase_extras["ifs_debug"] = ifs_debug
            phase_extras["phase_state"] = zf
            phase_extras["phase_gates"] = phase_gates
            phase_extras["learned_freqs"] = self.learned_freqs
        else:
            phase_extras = {
                "ifs_debug": ifs_debug,
                "phase_state": zf,
                "phase_gates": phase_gates,
                "learned_freqs": self.learned_freqs
            }

        return final_phase, final_mag, phase_extras, mag_extras
