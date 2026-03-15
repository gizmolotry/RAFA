from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
# BLACKWELL FIX: No transformers or baseline modules allowed
# from transformers import ClapTextModelWithProjection

from hf_local import resolve_hf_pretrained_path
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


# class BaselineDenoiser(nn.Module):
#     def __init__(self, freq_bins: int, tdim: int = 64):
#         super().__init__()
#         self.temb = TimeEmbedding(tdim)
#         in_ch = 3  # logmag + phase re/im
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch + 1, 32, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(32, 3, 3, padding=1),
#         )
# 
#     def forward(
#         self,
#         xt_mag: torch.Tensor,
#         xt_z: torch.Tensor,
#         t: torch.Tensor,
#         tokens: dict[str, torch.Tensor] | None = None,
#     ) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
#         # inputs [B,F,T], [B,F,T,2]
#         _ = tokens
#         bsz = xt_mag.size(0)
#         x = torch.stack([xt_mag, xt_z[..., 0], xt_z[..., 1]], dim=1)  # [B,3,F,T]
#         te = self.temb(t).mean(dim=1, keepdim=True).view(bsz, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
#         y = self.net(torch.cat([x, te], dim=1))
#         pred_mag = y[:, 0]
#         pred_z = phasor_normalize(torch.stack([y[:, 1], y[:, 2]], dim=-1))
#         return pred_mag, pred_z, {}, {}


class RAFADenoiser(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.temb = TimeEmbedding(64)
        self.time_scale = nn.Linear(64, 1)
        dcfg = cfg.get("diffusion", {})
        self.use_clap_cond = bool(dcfg.get("use_clap_cond", dcfg.get("use_clip_cond", False)))
        self.clap_load_error: str | None = None
        self.cond_scale = float(dcfg.get("cond_mag_scale", 0.1))
        self.text_condition_mode = str(dcfg.get("text_condition_mode", "film")).strip().lower()
        if self.text_condition_mode not in {"film", "ifs_control", "hybrid"}:
            self.text_condition_mode = "film"
        self.semantic_condition_mode = str(dcfg.get("semantic_condition_mode", "none")).strip().lower()
        if self.semantic_condition_mode not in {"none", "tension_envelope"}:
            self.semantic_condition_mode = "none"
        self.text_ifs_control_strength = float(dcfg.get("text_ifs_control_strength", 0.25))
        self.semantic_tension_strength = float(dcfg.get("semantic_tension_strength", 0.75))
        self.semantic_tension_alpha_strength = float(dcfg.get("semantic_tension_alpha_strength", 0.2))
        self.semantic_tension_temp_strength = float(dcfg.get("semantic_tension_temp_strength", 0.35))
        self.semantic_tension_delta_strength = float(dcfg.get("semantic_tension_delta_strength", 0.25))
        qset = cfg.get("phase_native_ifs", {}).get("qset", [2, 3, 4, 5, 6, 8, 12])
        self.semantic_qset = tuple(int(q) for q in qset)
        self.text_ifs_q_count = max(1, len(qset))
        default_harmonic = [2, 3, 4, 6, 8, 12]
        default_inharmonic = [q for q in self.semantic_qset if q not in default_harmonic]
        self.semantic_harmonic_qs = {
            int(q) for q in dcfg.get("semantic_tension_harmonic_qs", default_harmonic)
        }
        self.semantic_inharmonic_qs = {
            int(q) for q in dcfg.get("semantic_tension_inharmonic_qs", default_inharmonic)
        }
        self.text_encoder = None
        self.film = None
        self.ifs_text_head = None
        if self.use_clap_cond:
            try:
                text_encoder_id = resolve_hf_pretrained_path(
                    cfg["model"]["text_encoder"],
                    need_model=True,
                )
                self.text_encoder = ClapTextModelWithProjection.from_pretrained(
                    text_encoder_id,
                    local_files_only=os.path.isdir(text_encoder_id),
                )
                tdim = int(getattr(self.text_encoder.config, "projection_dim", self.text_encoder.config.hidden_size))
                if self.text_condition_mode in {"film", "hybrid"}:
                    self.film = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, 2))
                if self.text_condition_mode in {"ifs_control", "hybrid"}:
                    self.ifs_text_head = nn.Sequential(
                        nn.Linear(tdim, tdim),
                        nn.SiLU(),
                        nn.Linear(tdim, 3 + self.text_ifs_q_count),
                    )
                self.text_encoder.eval()
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
            except Exception as exc:
                self.use_clap_cond = False
                self.text_encoder = None
                self.film = None
                self.ifs_text_head = None
                self.clap_load_error = str(exc)
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

    def _tension_envelope(self, t_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if t_frames <= 1:
            return torch.zeros((1, 1, 1), device=device, dtype=dtype)
        pos = torch.linspace(0.0, 1.0, t_frames, device=device, dtype=dtype)
        # Triangle: 0 -> 1 -> 0.
        env = 1.0 - (2.0 * pos - 1.0).abs()
        return env.view(1, t_frames, 1)

    def _ifs_controls_from_tension(self, batch_size: int, t_frames: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor] | None:
        if self.semantic_condition_mode != "tension_envelope":
            return None
        env = self._tension_envelope(t_frames, device=device, dtype=dtype).expand(batch_size, -1, -1)
        strength = max(0.0, float(self.semantic_tension_strength))
        alpha_strength = max(0.0, float(self.semantic_tension_alpha_strength))
        temp_strength = max(0.0, float(self.semantic_tension_temp_strength))
        delta_strength = max(0.0, float(self.semantic_tension_delta_strength))

        q_bias = torch.ones((batch_size, t_frames, self.text_ifs_q_count), device=device, dtype=dtype)
        harmonic_gain = (1.0 - env) * strength
        inharmonic_gain = env * strength
        for idx, q in enumerate(self.semantic_qset):
            if q in self.semantic_harmonic_qs:
                q_bias[:, :, idx] = (1.0 + harmonic_gain.squeeze(-1)).clamp_min(1e-3)
            elif q in self.semantic_inharmonic_qs:
                q_bias[:, :, idx] = (1.0 + inharmonic_gain.squeeze(-1)).clamp_min(1e-3)
        q_bias = q_bias / q_bias.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        q_bias = q_bias * float(self.text_ifs_q_count)

        alpha = (1.0 + alpha_strength * (1.0 - env)).clamp_min(1e-3)
        temp = (1.0 + temp_strength * env).clamp_min(1e-3)
        delta = (1.0 + delta_strength * env).clamp_min(1e-3)
        return {
            "alpha_scale": alpha,
            "temp_scale": temp,
            "delta_scale": delta,
            "qset_scale": q_bias,
            "semantic_tension": env,
        }

    def _ifs_controls_from_text(self, txt: torch.Tensor, t_frames: int) -> dict[str, torch.Tensor] | None:
        if self.ifs_text_head is None:
            return None
        ctrl = self.ifs_text_head(txt)
        strength = max(0.0, float(self.text_ifs_control_strength))
        raw_alpha = ctrl[:, 0:1]
        raw_temp = ctrl[:, 1:2]
        raw_delta = ctrl[:, 2:3]
        raw_q = ctrl[:, 3 : 3 + self.text_ifs_q_count]
        ones = torch.ones_like(raw_alpha)
        alpha = ones + strength * torch.tanh(raw_alpha)
        temp = ones + strength * torch.tanh(raw_temp)
        delta = ones + strength * torch.tanh(raw_delta)
        q_nominal = torch.ones_like(raw_q)
        q_scaled = torch.softmax(raw_q, dim=-1) * float(self.text_ifs_q_count)
        qset = (1.0 - strength) * q_nominal + strength * q_scaled
        return {
            "alpha_scale": alpha.unsqueeze(1).expand(-1, t_frames, -1),
            "temp_scale": temp.unsqueeze(1).expand(-1, t_frames, -1),
            "delta_scale": delta.unsqueeze(1).expand(-1, t_frames, -1),
            "qset_scale": qset.unsqueeze(1).expand(-1, t_frames, -1),
        }

    def forward(
        self,
        xt_mag: torch.Tensor,
        xt_z: torch.Tensor,
        t: torch.Tensor,
        tokens: dict[str, torch.Tensor] | None = None,
        semantic_controls: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
        phase = phasor_to_phase(xt_z)
        xmag = xt_mag
        ifs_text_controls = None
        # Use provided semantic_controls (from GT tension) or generate from tension_envelope
        sc = semantic_controls if semantic_controls is not None else self._ifs_controls_from_tension(
            batch_size=phase.size(0),
            t_frames=phase.size(-1),
            device=phase.device,
            dtype=phase.dtype,
        )
        if self.use_clap_cond and tokens is not None:
            self.text_encoder.eval()
            with torch.no_grad():
                txt = self.text_encoder(
                    input_ids=tokens.get("input_ids"),
                    attention_mask=tokens.get("attention_mask"),
                ).text_embeds
            if self.film is not None:
                gb = self.film(txt)
                gamma = gb[:, 0].view(-1, 1, 1)
                beta = gb[:, 1].view(-1, 1, 1)
                # Magnitude-only text path; used in film/hybrid modes.
                xmag = xmag * (1.0 + self.cond_scale * gamma) + self.cond_scale * beta
            ifs_text_controls = self._ifs_controls_from_text(txt, t_frames=phase.size(-1))
        pred_phase, pred_mag, phase_extras, mag_extras = self.rafa(
            xmag,
            phase,
            None,
            ifs_text_controls=ifs_text_controls,
            semantic_controls=sc,
        )
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
        if ifs_text_controls is not None:
            phase_extras["ifs_text_controls"] = ifs_text_controls
        if sc is not None:
            phase_extras["semantic_controls"] = sc

        return pred_mag, pred_z, phase_extras, mag_extras
