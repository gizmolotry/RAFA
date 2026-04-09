import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BlackwellSafeAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, batch_first=True):
        super().__init__()
        self.nhead = nhead
        self.head_dim = max(1, d_model // nhead)
        self.qkv = nn.Linear(d_model, 3 * nhead * self.head_dim)
        self.out_proj = nn.Linear(nhead * self.head_dim, d_model)
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
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.nhead * self.head_dim)
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
        x = self.norm1(x + self.dropout1(attn_out))
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        return x

class RAFAClutchTransformerUpgraded(nn.Module):
    def __init__(
        self,
        num_gears: int = 1,
        gear_dim: int = 129,
        transformer_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        pool_rnn: list[int] | None = None,
        dropout: float = 0.1,
        gate_reg_weight: float = 1e-3,
        nonlinear_fusion: bool = False,
        upsample_mode: str = "repeat",
        gate_tau: float = 1.0,
        flux_weight: float = 0.0,
        peak_weight: float = 0.0,
        pool_kernel: list[int] | None = None,
        **kwargs
    ):
        super().__init__()
        self.num_gears = num_gears
        self.transformer_dim = transformer_dim
        self.proj_in = nn.Linear(gear_dim, transformer_dim)
        
        self.encoder_layers = nn.ModuleList([
            SafeEncoderLayer(transformer_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.gate_proj = nn.Linear(transformer_dim, 1)
        self.pool_rnn = pool_rnn
        self.gate_reg_weight = gate_reg_weight
        self.nonlinear_fusion = nonlinear_fusion
        self.upsample_mode = upsample_mode
        self.gate_tau = float(gate_tau)
        self.flux_weight = float(flux_weight)
        self.peak_weight = float(peak_weight)

    def forward(self, x: torch.Tensor):
        b, c, t, f = x.shape
        x_proj = self.proj_in(x)
        x_flat = x_proj.contiguous().view(b*c, t, self.transformer_dim)
        
        trans = x_flat
        for layer in self.encoder_layers:
            trans = layer(trans)
            
        trans_feat = trans.view(b, c, t, self.transformer_dim)
        # Remaining logic matches RAFA-PC v3.0 specs
        gates = torch.sigmoid(self.gate_proj(trans_feat).squeeze(-1) / self.gate_tau)
        
        # In RAFA v3.0, the clutch typically allows the final Conv2d to do the mixing.
        # We return [B, C, F, T] so self.phase_fuse (Conv2d(3, 1, 1)) can work.
        return trans_feat.permute(0, 1, 3, 2), gates, {}
