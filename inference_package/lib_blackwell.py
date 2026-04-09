import torch
import torch.nn.functional as F
import math
from triton_kernels import ramanujan_summary_triton

# --- RAFA-M (80M) FULL-POWER TRITON-FFT CORE ---
# Scale: Voice 45M | Spine 25M | Brain 10M

def init_weight(shape, dev='cuda'):
    w = torch.empty(*shape, device=dev, dtype=torch.float32)
    if len(shape) >= 2:
        torch.nn.init.xavier_uniform_(w)
    else:
        torch.nn.init.zeros_(w)
    w.requires_grad = True
    return w

class NakedDenoiser:
    def __init__(self, dev='cuda'):
        self.weights = {
            "conv_in_w": init_weight((128, 3, 3, 3), dev),
            "conv_in_b": init_weight((128,), dev),
            "cond_proj_w": init_weight((128, 3, 1, 1), dev),
            "cond_proj_b": init_weight((128,), dev),
            "cond_gate_w": init_weight((128, 1, 1, 1), dev),
            "stage1_w": init_weight((256, 128, 3, 3), dev),
            "stage1_b": init_weight((256,), dev),
            "stage2_w": init_weight((512, 256, 3, 3), dev),
            "stage2_b": init_weight((512,), dev),
            "stage3_w": init_weight((512, 512, 3, 3), dev),
            "stage3_b": init_weight((512,), dev),
            "stage4_w": init_weight((512, 512, 3, 3), dev),
            "stage4_b": init_weight((512,), dev),
            "conv_out_w": init_weight((3, 512, 3, 3), dev),
            "conv_out_b": init_weight((3,), dev)
        }

    def forward(self, xt_mag, xt_z, t, rafa_mag=None, rafa_z=None):
        x = torch.stack([xt_mag, xt_z[..., 0], xt_z[..., 1]], dim=1) 
        h = F.relu(F.conv2d(x, self.weights["conv_in_w"], self.weights["conv_in_b"], padding=1))
        if rafa_z is not None:
            guide = torch.stack([rafa_z[..., 0], rafa_z[..., 1]], dim=1)
            guide = (guide - guide.mean(dim=(2,3), keepdim=True)) / (guide.std(dim=(2,3), keepdim=True) + 1e-6)
            guide_flat = guide.permute(0, 2, 3, 1)
            guide_feat = F.linear(guide_flat, self.weights["cond_proj_w"][:, :2].view(128, 2), self.weights["cond_proj_b"]).permute(0, 3, 1, 2)
            h = h + torch.sigmoid(self.weights["cond_gate_w"].view(1, 128, 1, 1)) * guide_feat
        h = F.relu(F.conv2d(h, self.weights["stage1_w"], self.weights["stage1_b"], padding=1))
        h = F.relu(F.conv2d(h, self.weights["stage2_w"], self.weights["stage2_b"], padding=1))
        h = F.relu(F.conv2d(h, self.weights["stage3_w"], self.weights["stage3_b"], padding=1))
        h = F.relu(F.conv2d(h, self.weights["stage4_w"], self.weights["stage4_b"], padding=1))
        y = F.conv2d(h, self.weights["conv_out_w"], self.weights["conv_out_b"], padding=1)
        pred_mag = y[:, 0]
        pz = torch.stack([y[:, 1], y[:, 2]], dim=-1)
        pred_z = pz / torch.sqrt((pz*pz).sum(-1, keepdim=True).clamp_min(1e-8))
        return pred_mag, pred_z

class NakedRAFA:
    def __init__(self, d_model=768, freq_bins=129, dev='cuda'):
        self.d_model = d_model
        self.freq_bins = freq_bins
        h_dim = 1024
        self.weights = {
            "lexicon_phasors": init_weight((256, freq_bins * 2), dev),
            "mode_seeds": init_weight((10, freq_bins, 2), dev),
            "spectrum_biases": init_weight((4, freq_bins), dev),
            "sp_w_proj": init_weight((3 * d_model, 6 * freq_bins), dev),
            "sp_b_proj": init_weight((3 * d_model,), dev),
            "sp_filter_w": init_weight((d_model, 1, 251), dev),
            "sp_w_out": init_weight((d_model, d_model), dev),
            "sp_b_out": init_weight((d_model,), dev),
            "g1_w_ph": init_weight((freq_bins, d_model), dev),
            "g1_w_mag": init_weight((freq_bins, d_model), dev),
            "ps_w_ih": init_weight((3 * h_dim, freq_bins), dev),
            "ps_w_hh": init_weight((3 * h_dim, h_dim), dev),
            "ps_b_ih": init_weight((3 * h_dim,), dev),
            "ps_b_hh": init_weight((3 * h_dim,), dev),
            "ps_slow_proj": init_weight((freq_bins * 3, h_dim), dev),
            "ps_z0": init_weight((1, freq_bins, 2), dev),
        }
        self.qset = torch.tensor([2, 3, 4, 5, 6, 8, 12], dtype=torch.int32, device=dev)
        self.qw = torch.tensor([1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=torch.float32, device=dev)

    def forward(self, mag, phase, control_matrix=None, text_tokens=None, num_steps=4):
        B, F_dim, T = mag.shape
        w = self.weights
        if control_matrix and "spectrum_id" in control_matrix and control_matrix["spectrum_id"] is not None:
            s_idx = control_matrix["spectrum_id"]; s_bias = torch.sigmoid(w["spectrum_biases"][s_idx]).unsqueeze(-1) if not isinstance(s_idx, int) else torch.sigmoid(w["spectrum_biases"][s_idx]).view(1, F_dim, 1)
            mag = mag * s_bias
        x_raw = torch.cat([mag]*6, dim=1).transpose(1, 2)
        qkv = F.linear(x_raw, w["sp_w_proj"], w["sp_b_proj"])
        x_e, v_e, gate_e = qkv.chunk(3, dim=-1)
        L = T + 251 - 1; n_fft = 2**math.ceil(math.log2(L))
        X_e = torch.fft.rfft(x_e.transpose(1, 2), n=n_fft); W_e = torch.fft.rfft(w["sp_filter_w"].squeeze(1), n=n_fft)
        x_filt = torch.fft.irfft(X_e * W_e.unsqueeze(0), n=n_fft)[..., :T]
        spine_out = (x_filt.transpose(1, 2) * v_e) * torch.sigmoid(gate_e)
        spine_feat = F.linear(spine_out, w["sp_w_out"], w["sp_b_out"])
        p_seed = F.linear(spine_feat, w["g1_w_ph"]).transpose(1, 2)
        m_seed = F.linear(spine_feat, w["g1_w_mag"]).transpose(1, 2)
        if control_matrix and "mode_id" in control_matrix and control_matrix["mode_id"] is not None:
            m_idx = control_matrix["mode_id"]; z_prev = _renorm(w["mode_seeds"][m_idx]).expand(B, -1, -1) if isinstance(m_idx, int) else _renorm(w["mode_seeds"][m_idx])
        else:
            z_prev = _renorm(w["ps_z0"]).expand(B, -1, -1)
        z_in = torch.stack([torch.cos(p_seed), torch.sin(p_seed)], dim=-1).permute(0, 2, 1, 3)
        h_slow = z_in.new_zeros(B, 1024)
        h_slow_list = []
        outs = []
        for t in range(T):
            f_v = ramanujan_summary_triton(z_in[:, t], self.qset, self.qw).mean(dim=1)
            ih, hh = F.linear(f_v, w["ps_w_ih"][:, :5], w["ps_b_ih"]), F.linear(h_slow, w["ps_w_hh"], w["ps_b_hh"])
            i_r, i_z, i_n = ih.chunk(3, dim=-1); h_r, h_z, h_n = hh.chunk(3, dim=-1)
            r, z_gate = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
            h_slow = (1.0 - z_gate) * torch.tanh(i_n + r * h_n) + z_gate * h_slow
            h_slow_list.append(h_slow)
            brain_ctrl = F.linear(h_slow, w["ps_slow_proj"]).view(B, self.freq_bins, 3)
            brain_delta = 0.5 * torch.tanh(brain_ctrl[:, :, :2])
            brain_z = torch.stack([torch.cos(brain_delta[..., 0]), torch.sin(brain_delta[..., 1])], dim=-1)
            coupling_gate = torch.sigmoid(brain_ctrl[:, :, 2:3])
            z_iter = (1.0 - coupling_gate) * z_in[:, t] + coupling_gate * z_prev
            for _ in range(num_steps):
                z_iter_re = z_iter[..., 0] * brain_z[..., 0] - z_iter[..., 1] * brain_z[..., 1]
                z_iter_im = z_iter[..., 0] * brain_z[..., 1] + z_iter[..., 1] * brain_z[..., 0]
                z_iter = torch.stack([z_iter_re, z_iter_im], dim=-1)
                z_iter = z_iter + 0.1 * brain_z
                z_iter = _renorm(z_iter)
            z_prev = z_iter; outs.append(z_iter)
        zf = torch.stack(outs, dim=1).permute(0, 2, 1, 3)
        return torch.atan2(zf[..., 1], zf[..., 0] + 1e-12), m_seed, {"phase_state": zf, "h_slow": torch.stack(h_slow_list, dim=1)}

def _renorm(z, eps=1e-8):
    n = torch.sqrt((z * z).sum(dim=-1, keepdim=True).clamp_min(eps))
    return z / n
