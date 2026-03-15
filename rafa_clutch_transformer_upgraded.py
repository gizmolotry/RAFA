# rafa_clutch_transformer_upgraded.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from rafa_math_tools import geodesic_phase_upsample

class BlackwellSafeAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, batch_first=True):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        if self.batch_first:
            B, T, C = x.shape
        else:
            T, B, C = x.shape
            x = x.transpose(0, 1)
            
        qkv = self.qkv(x).view(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Transpose for [B, nhead, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, attn

class SafeEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = BlackwellSafeAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

class RAFAClutchTransformerUpgraded(nn.Module):
    def __init__(
        self,
        num_gears: int,
        gear_dim: int,
        transformer_dim: int,
        num_layers: int,
        num_heads: int,
        pool_kernel: list[int],
        dropout: float,
        gate_reg_weight: float,
        nonlinear_fusion: bool,
        upsample_mode: str,
        gate_tau: float = 1.0,
        flux_weight: float = 0.0,
        peak_weight: float = 0.0,
    ):
        super().__init__()
        self.num_gears = num_gears
        self.transformer_dim = transformer_dim
        self.proj_in = nn.Linear(gear_dim, transformer_dim)
        
        # BLACKWELL FIX: Use a pure PyTorch manual transformer stack
        self.encoder_layers = nn.ModuleList([
            SafeEncoderLayer(transformer_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.gate_proj = nn.Linear(transformer_dim, 1)
        self.pool_kernel = pool_kernel
        self.gate_reg_weight = gate_reg_weight
        self.nonlinear_fusion = nonlinear_fusion
        self.upsample_mode = upsample_mode
        self.gate_tau = float(gate_tau)
        self.flux_weight = float(flux_weight)
        self.peak_weight = float(peak_weight)

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape (batch, channels=num_gears, time, features=gear_dim)
        returns: fused output, gates, extras dict
        """
        b, c, t, f = x.shape
        x_proj = self.proj_in(x) # (b, c, t, transformer_dim)

        x_flat = x_proj.contiguous().view(b*c, t, self.transformer_dim)
        
        # Manual encoder forward pass
        trans = x_flat
        for layer in self.encoder_layers:
            trans = layer(trans)
            
        trans_feat = trans.view(b, c, t, self.transformer_dim)

        # Gate summary includes transient sensitivity via temporal flux.
        mean_feat = trans_feat.mean(2)  # (b,c,d)
        if t > 1:
            flux_feat = (trans_feat[:, :, 1:, :] - trans_feat[:, :, :-1, :]).abs().mean(2)
        else:
            flux_feat = torch.zeros_like(mean_feat)
        peak_feat = trans_feat.abs().amax(dim=2)
        gate_summary = mean_feat + self.flux_weight * flux_feat + self.peak_weight * peak_feat

        gate_logits = self.gate_proj(gate_summary).squeeze(-1)  # (b,c)
        gates = torch.softmax(gate_logits / max(self.gate_tau, 1e-6), dim=1)  # (b,c)

        # Path B: Residual Gating (gate update, not gate state)
        # We ensure gears are never fully muted by adding the gates as a residual boost.
        gated_x = x * (1.0 + gates.unsqueeze(-1).unsqueeze(-1))
        fused = gated_x

        if self.nonlinear_fusion:
            fused = torch.tanh(fused)

        if self.upsample_mode == "repeat":
            # Keep output shape invariant with input shape.
            # The prior implementation expanded one axis and broke downstream diffusion dimensions.
            fused = fused
        
        extras = {"gate_reg_loss": (gates**2).mean() * self.gate_reg_weight}
        return fused, gates, extras

# The rest of this file (below) mirrors parts of model.py and is included for completeness,
# but the core RAFA modules above are identical to those used in model.py.
# If you modify the transformer logic, you can do so here and keep model.py clean.
