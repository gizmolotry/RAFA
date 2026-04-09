import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_utils import phasor_normalize

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, t):
        # t: [B]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.size(1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(1)))
        return self.proj(emb)

class BaselineDenoiser(nn.Module):
    def __init__(self, freq_bins):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
    def forward(self, x_mag, x_z, t, tokens=None):
        return x_mag, x_z, {}, {}

class RAFADenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # BLACKWELL sm_120 COMPATIBLE RENDERER
        self.temb = TimeEmbedding(64)
        self.time_scale = nn.Linear(64, 1)
        
        # We assume the input features are already provided by the RAFA core
        # or ground truth latents during Stage 1.
        
        # Simple U-Net like structure for Phase/Mag reconstruction
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1) # mag + phase_re + phase_im
        self.blocks = nn.ModuleList([
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1)
        ])
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, xt_mag, xt_z, t, rafa_mag=None, rafa_z=None):
        # xt_mag: [B, F, T]
        # xt_z: [B, F, T, 2]
        # rafa_mag/z: same shapes, provided by core
        
        bsz = xt_mag.size(0)
        x = torch.stack([xt_mag, xt_z[..., 0], xt_z[..., 1]], dim=1) # [B, 3, F, T]
        
        # Inject RAFA core features if provided
        if rafa_mag is not None and rafa_z is not None:
            rafa_feat = torch.stack([rafa_mag, rafa_z[..., 0], rafa_z[..., 1]], dim=1)
            x = x + rafa_feat
            
        h = self.conv_in(x)
        for b in self.blocks:
            h = b(h)
            
        y = self.conv_out(h)
        pred_mag = y[:, 0]
        pred_z = phasor_normalize(torch.stack([y[:, 1], y[:, 2]], dim=-1))
        
        return pred_mag, pred_z, {}, {}
