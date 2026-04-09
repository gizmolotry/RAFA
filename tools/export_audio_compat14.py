import argparse
import json
import math
import os
import wave
import random

import numpy as np
import torch
import torch.nn.functional as F

try:
    from lib_blackwell import NakedDenoiser
except ImportError:
    from core.lib_blackwell import NakedDenoiser
try:
    from dataset import RafaGuerrillaDataset
except ImportError:
    from core.dataset import RafaGuerrillaDataset
from config import load_config
from diffusion_utils import make_beta_schedule, phase_to_phasor, phasor_to_phase, q_sample_x0
from triton_kernels import ramanujan_summary_triton
import stft_utils


def _renorm(z, eps=1e-8):
    return z / torch.sqrt((z * z).sum(dim=-1, keepdim=True).clamp_min(eps))


def save_wav(mag, phase, path, stft_cfg, sr=16000):
    m = torch.as_tensor(mag).cpu()
    p = torch.as_tensor(phase).cpu()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if p.ndim == 2:
        p = p.unsqueeze(0)
    wav = stft_utils.inverse_stft(m, p, stft_cfg).squeeze(0).cpu().numpy()
    wav = wav / (np.abs(wav).max() + 1e-8)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((wav * 32767).astype(np.int16).tobytes())


def load_wav(path):
    with wave.open(path, "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32767.0


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Compat14RAFA:
    """Compatibility runtime for 14-key stage-4 checkpoints.

    This path is intentionally isolated from the live lib_blackwell runtime.
    It reconstructs the missing g1 phase/magnitude heads from the older
    14-key checkpoint family using an empirically tuned fallback.
    """

    def __init__(self, weights, dev="cuda", phase_mix=0.395, mag_mix=0.05, source="qkv", start_idx=516):
        self.weights = weights
        self.dev = dev
        self.phase_mix = float(phase_mix)
        self.mag_mix = float(mag_mix)
        self.source = str(source)
        self.start_idx = int(start_idx)
        self.qset = torch.tensor([2, 3, 4, 5, 6, 8, 12], dtype=torch.int32, device=dev)
        self.qw = torch.tensor([1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=torch.float32, device=dev)

    def _seed_features(self, mag):
        w = self.weights
        x_raw = torch.cat([mag] * 6, dim=1).transpose(1, 2)
        qkv = F.linear(x_raw, w["sp_w_proj"], w["sp_b_proj"])
        x_e, v_e, gate_e = qkv.chunk(3, dim=-1)
        t_len = mag.size(-1)
        filt_len = w["sp_filter_w"].size(-1)
        n_fft = 2 ** math.ceil(math.log2(t_len + filt_len - 1))
        x_fft = torch.fft.rfft(x_e.transpose(1, 2), n=n_fft)
        w_fft = torch.fft.rfft(w["sp_filter_w"].squeeze(1), n=n_fft)
        x_filt = torch.fft.irfft(x_fft * w_fft.unsqueeze(0), n=n_fft)[..., :t_len]
        spine_out = (x_filt.transpose(1, 2) * v_e) * torch.sigmoid(gate_e)
        spine_feat = F.linear(spine_out, w["sp_w_out"], w["sp_b_out"])
        return x_e, v_e, x_filt, spine_feat

    def _project_legacy_heads(self, mag, phase):
        bsz, freq_bins, _ = mag.shape
        x_e, v_e, x_filt, spine_feat = self._seed_features(mag)
        
        start = self.start_idx
        end = start + freq_bins
        
        if self.source == "spine":
            base_p = spine_feat[..., start:end].transpose(1, 2)
            base_m = spine_feat[..., end : end + freq_bins].transpose(1, 2) if spine_feat.size(-1) >= end + freq_bins else spine_feat[..., start:end].transpose(1, 2)
        elif self.source == "qkv":
            base_p = x_e[..., start:end].transpose(1, 2)
            base_m = v_e[..., start:end].transpose(1, 2)
        elif self.source == "filt":
            base_p = x_filt[:, start:end, :]
            base_m = x_filt[:, end : end + freq_bins, :] if x_filt.size(1) >= end + freq_bins else x_filt[:, start:end, :]
        else:
            raise ValueError(f"Unknown compatibility source: {self.source}")
            
        p_seed = phase + self.phase_mix * base_p
        m_seed = mag + self.mag_mix * base_m
        return p_seed, m_seed

    def forward(self, mag, phase, control_matrix=None, num_steps=4):
        bsz, freq_bins, t_len = mag.shape
        w = self.weights
        if control_matrix and control_matrix.get("spectrum_id") is not None and "spectrum_biases" in w:
            spectrum_id = int(control_matrix["spectrum_id"])
            mag = mag * torch.sigmoid(w["spectrum_biases"][spectrum_id]).view(1, freq_bins, 1)

        p_seed, m_seed = self._project_legacy_heads(mag, phase)
        mode_id = int(control_matrix["mode_id"]) if control_matrix and control_matrix.get("mode_id") is not None else 0
        z_prev = _renorm(w["mode_seeds"][mode_id]).expand(bsz, -1, -1)
        z_in = torch.stack([torch.cos(p_seed), torch.sin(p_seed)], dim=-1).permute(0, 2, 1, 3)

        h_slow = z_in.new_zeros(bsz, 1024)
        outs = []
        h_slow_list = []
        for t in range(t_len):
            f_v = ramanujan_summary_triton(z_in[:, t], self.qset, self.qw).mean(dim=1)
            ih = F.linear(f_v, w["ps_w_ih"][:, :5], w["ps_b_ih"])
            hh = F.linear(h_slow, w["ps_w_hh"], w["ps_b_hh"])
            i_r, i_z, i_n = ih.chunk(3, dim=-1)
            h_r, h_z, h_n = hh.chunk(3, dim=-1)
            r_gate = torch.sigmoid(i_r + h_r)
            z_gate = torch.sigmoid(i_z + h_z)
            h_slow = (1.0 - z_gate) * torch.tanh(i_n + r_gate * h_n) + z_gate * h_slow
            h_slow_list.append(h_slow)

            brain_ctrl = F.linear(h_slow, w["ps_slow_proj"]).view(bsz, freq_bins, 3)
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
            z_prev = z_iter
            outs.append(z_iter)

        zf = torch.stack(outs, dim=1).permute(0, 2, 1, 3)
        return torch.atan2(zf[..., 1], zf[..., 0] + 1e-12), m_seed, {"phase_state": zf, "h_slow": torch.stack(h_slow_list, dim=1)}


def render_graduation_suite_compat14(ckpt_path, out_dir, phase_mix=0.395, mag_mix=0.05, source="qkv", seed=101, start_idx=516):
    cfg = load_config()
    dev = torch.device("cuda")
    stft_cfg = cfg["data"]["stft"]
    os.makedirs(out_dir, exist_ok=True)
    set_deterministic_seed(int(seed))

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    core_weights = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ckpt["core"].items()}
    denoiser = NakedDenoiser(dev=dev)
    for k, v in ckpt["denoiser"].items():
        if k in denoiser.weights:
            denoiser.weights[k] = v.to(dev)
    rafa_core = Compat14RAFA(core_weights, dev=dev, phase_mix=phase_mix, mag_mix=mag_mix, source=source, start_idx=start_idx)

    ds = RafaGuerrillaDataset(cfg, split="training")
    suite = [
        (42, "engine", 1),
        (100, "voice", 2),
        (0, "impact", 3),
        (200, "drone", 4),
    ]
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    t = torch.ones(1, dtype=torch.long, device=dev) * 25

    rows = []
    with torch.no_grad():
        for idx, name, mode_id in suite:
            mag, phase, _, _, _ = ds[idx]
            mag, phase = mag.to(dev).unsqueeze(0), phase.to(dev).unsqueeze(0)
            x0_mag, x0_z = torch.log1p(mag), phase_to_phasor(phase)
            xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)

            control_matrix = {"mode_id": mode_id}
            if "spectrum_biases" in core_weights:
                control_matrix["spectrum_id"] = 1
            p_p, p_m, _ = rafa_core.forward(mag, phase, control_matrix=control_matrix)
            d_mag, d_z = denoiser.forward(
                xt_mag,
                xt_z,
                t,
                rafa_mag=p_m,
                rafa_z=torch.stack([torch.cos(p_p), torch.sin(p_p)], -1),
            )

            out_path = os.path.join(out_dir, f"grad_{name}.wav")
            save_wav(torch.expm1(d_mag[0]), phasor_to_phase(d_z)[0], out_path, stft_cfg)
            rows.append({"name": name, "out_path": out_path, "mode_id": mode_id})
            print(f"RENDERED: {out_path}")

    meta = {
        "checkpoint": ckpt_path,
        "out_dir": out_dir,
        "phase_mix": phase_mix,
        "mag_mix": mag_mix,
        "source": source,
        "seed": int(seed),
        "start_idx": int(start_idx),
        "renders": rows,
    }
    meta_path = os.path.join(out_dir, "compat14_render_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"WROTE: {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_stage4/bound_weights_ep10_step1200.pt")
    ap.add_argument("--out-dir", default="outputs/graduation_pack_restore_candidate_refined")
    ap.add_argument("--phase-mix", type=float, default=0.395)
    ap.add_argument("--mag-mix", type=float, default=0.05)
    ap.add_argument("--source", default="qkv", choices=["spine", "qkv", "filt"])
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--start-idx", type=int, default=516)
    args = ap.parse_args()
    render_graduation_suite_compat14(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        phase_mix=args.phase_mix,
        mag_mix=args.mag_mix,
        source=args.source,
        seed=args.seed,
        start_idx=args.start_idx,
    )


if __name__ == "__main__":
    main()
