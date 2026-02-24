# rafa_clutch_transformer_upgraded.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from rafa_math_tools import geodesic_phase_upsample

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
        self.transformer = nn.Transformer(
            d_model=transformer_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_proj = nn.Linear(transformer_dim, 1) # Changed from num_gears to 1
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

        # Reshape for transformer: treat each gear's time series as a batch item
        x_flat = x_proj.contiguous().view(b*c, t, self.transformer_dim)

        trans = self.transformer.encoder(x_flat) # (b*c, t, transformer_dim)
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

        # Apply gates to the original input features
        gated_x = x * gates.unsqueeze(-1).unsqueeze(-1)
        fused = gated_x

        if self.nonlinear_fusion:
            fused = torch.tanh(fused)

        if self.upsample_mode == "repeat":
            # For phase data, linear repetition breaks winding numbers. Use Geodesic Upsampling.
            if fused.shape[1] == 1: # Single channel, assumed to be phase if complex
                # We need to infer if this is phase or magnitude.
                # Heuristic: if values are unbounded, likely mag. If wrapped, phase.
                # Safer: assume phase upsampling is always better for continuous signals.
                fused = geodesic_phase_upsample(fused.squeeze(1), scale_factor=self.pool_kernel[0], mode='linear').unsqueeze(1)
            else:
                fused = fused.repeat(1, 1, 1, 1)
        
        extras = {"gate_reg_loss": (gates**2).mean() * self.gate_reg_weight}
        return fused, gates, extras

# The rest of this file (below) mirrors parts of model.py and is included for completeness,
# but the core RAFA modules above are identical to those used in model.py.
# If you modify the transformer logic, you can do so here and keep model.py clean.
