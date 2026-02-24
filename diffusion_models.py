from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ClapTextModelWithProjection

from model import RAFA
from diffusion_utils import phasor_to_phase, phase_to_phasor, phasor_normalize


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / max(1, half - 1)))
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if emb.size(1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(1)))
        return self.proj(emb)


class BaselineDenoiser(nn.Module):
    def __init__(self, freq_bins: int, tdim: int = 64):
        super().__init__()
        self.temb = TimeEmbedding(tdim)
        in_ch = 3  # logmag + phase re/im
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + 1, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(
        self,
        xt_mag: torch.Tensor,
        xt_z: torch.Tensor,
        t: torch.Tensor,
        tokens: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
        # inputs [B,F,T], [B,F,T,2]
        _ = tokens
        bsz = xt_mag.size(0)
        x = torch.stack([xt_mag, xt_z[..., 0], xt_z[..., 1]], dim=1)  # [B,3,F,T]
        te = self.temb(t).mean(dim=1, keepdim=True).view(bsz, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        y = self.net(torch.cat([x, te], dim=1))
        pred_mag = y[:, 0]
        pred_z = phasor_normalize(torch.stack([y[:, 1], y[:, 2]], dim=-1))
        return pred_mag, pred_z, {}, {}


class RAFADenoiser(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.temb = TimeEmbedding(64)
        self.time_scale = nn.Linear(64, 1)
        dcfg = cfg.get("diffusion", {})
        self.use_clap_cond = bool(dcfg.get("use_clap_cond", dcfg.get("use_clip_cond", False)))
        self.cond_scale = float(dcfg.get("cond_mag_scale", 0.1))
        self.text_encoder = None
        self.film = None
        if self.use_clap_cond:
            try:
                self.text_encoder = ClapTextModelWithProjection.from_pretrained(cfg["model"]["text_encoder"])
                tdim = int(getattr(self.text_encoder.config, "projection_dim", self.text_encoder.config.hidden_size))
                self.film = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, 2))
                self.text_encoder.eval()
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
            except Exception:
                self.use_clap_cond = False
                self.text_encoder = None
                self.film = None
        rcfg = dict(cfg)
        rcfg["phase_native_ifs"] = dict(cfg.get("phase_native_ifs", {}))
        if "rafa_solver_enabled" in dcfg:
            rcfg["phase_native_ifs"]["enabled"] = bool(dcfg.get("rafa_solver_enabled", True))
        if "rafa_solver_num_steps" in dcfg and dcfg.get("rafa_solver_num_steps", None) is not None:
            rcfg["phase_native_ifs"]["num_steps"] = int(dcfg.get("rafa_solver_num_steps"))
        if "rafa_solver_time_stride" in dcfg and dcfg.get("rafa_solver_time_stride", None) is not None:
            rcfg["phase_native_ifs"]["time_stride"] = int(dcfg.get("rafa_solver_time_stride"))
        self.rafa = RAFA(rcfg)
        
        # Crystal Prediction Head (Idea 5): Predict the Coherence Matrix A [F, F]
        # We predict a flattened representation or a low-rank factor
        freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
        self.crystal_head = nn.Sequential(
            nn.Linear(freq_bins, 64),
            nn.SiLU(),
            nn.Linear(64, freq_bins) # Predicting a diagonal or row-sum for now to save params
        )

    def forward(
        self,
        xt_mag: torch.Tensor,
        xt_z: torch.Tensor,
        t: torch.Tensor,
        tokens: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
        phase = phasor_to_phase(xt_z)
        xmag = xt_mag
        if self.use_clap_cond and tokens is not None:
            self.text_encoder.eval()
            with torch.no_grad():
                txt = self.text_encoder(
                    input_ids=tokens.get("input_ids"),
                    attention_mask=tokens.get("attention_mask"),
                ).text_embeds
            gb = self.film(txt)
            gamma = gb[:, 0].view(-1, 1, 1)
            beta = gb[:, 1].view(-1, 1, 1)
            # Phase-safe conditioning: modulate magnitudes only.
            xmag = xmag * (1.0 + self.cond_scale * gamma) + self.cond_scale * beta
        pred_phase, pred_mag, phase_extras, mag_extras = self.rafa(xmag, phase, None)
        s = torch.sigmoid(self.time_scale(self.temb(t))).view(-1, 1, 1)
        pred_mag = (1.0 - s) * xmag + s * pred_mag
        # Blend directly in phasor space to avoid branch-cut artifacts around +/-pi.
        pred_phase_z = phase_to_phasor(pred_phase)
        sz = s.view(-1, 1, 1, 1)
        pred_z = phasor_normalize((1.0 - sz) * xt_z + sz * pred_phase_z)
        
        # Crystal Coherence Prediction (Symmetry Denoising)
        # We project the latent features or the output mag to predict the structural coherence
        pred_coh = self.crystal_head(pred_mag.mean(dim=2)) # [B, F]
        phase_extras["pred_coherence"] = pred_coh
        
        return pred_mag, pred_z, phase_extras, mag_extras
