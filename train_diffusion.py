from __future__ import annotations

import os
import time
import argparse
import re
import shutil
import json
import random
import hashlib
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import AutoTokenizer

from config import load_config
from dataset import RafaGuerrillaDataset
from diffusion_models import BaselineDenoiser, RAFADenoiser
from diffusion_utils import make_beta_schedule, extract, phase_to_phasor, q_sample_x0, phasor_normalize, phasor_to_phase, spectral_centroid
import rafa_math_tools as rmt


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float((g * g).sum().item())
    return total ** 0.5


def _preload_dataset(ds: RafaGuerrillaDataset) -> TensorDataset:
    mags = []
    phases = []
    labels = []
    for i in range(len(ds)):
        m, p, y = ds[i]
        mags.append(m)
        phases.append(p)
        labels.append(y)
    return TensorDataset(torch.stack(mags, dim=0), torch.stack(phases, dim=0), torch.stack(labels, dim=0))


def _checkpoint_epoch(path: str) -> int | None:
    m = re.search(r"diff_ep(\d+)\.pt$", os.path.basename(path))
    if m is None:
        return None
    return int(m.group(1))


def _prune_checkpoints(ckpt_dir: str, keep_last: int) -> None:
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.startswith("diff_ep") and f.endswith(".pt")]
    tagged = []
    for p in files:
        ep = _checkpoint_epoch(p)
        if ep is not None:
            tagged.append((ep, p))
    tagged.sort(key=lambda x: x[0])
    to_delete = max(0, len(tagged) - keep_last)
    for _, p in tagged[:to_delete]:
        try:
            os.remove(p)
        except OSError:
            pass


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _cfg_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _dataset_fingerprint(ds: RafaGuerrillaDataset) -> dict[str, object]:
    if bool(getattr(ds, "use_local_files", False)):
        entries = []
        paths = []
        folder_counts: dict[str, int] = {}
        for p in sorted(getattr(ds, "local_files", []), key=lambda x: str(x).lower()):
            pstr = p.as_posix()
            paths.append(pstr)
            folder = p.parent.as_posix()
            folder_counts[folder] = folder_counts.get(folder, 0) + 1
            try:
                st = p.stat()
                entries.append(f"{pstr}|{int(st.st_size)}|{int(st.st_mtime_ns)}")
            except OSError:
                entries.append(f"{pstr}|missing")
        h = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
        path_hash = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()
        return {
            "source": "local_wavs",
            "num_files": len(entries),
            "hash": h,
            "path_hash": path_hash,
            "folder_counts": dict(sorted(folder_counts.items())),
        }

    walker = getattr(getattr(ds, "ds", None), "_walker", None)
    if walker is not None:
        items = [str(x) for x in walker]
        h = hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()
        return {"source": "gtzan", "num_items": len(items), "hash": h}

    h = hashlib.sha256(f"len={len(ds)}".encode("utf-8")).hexdigest()
    return {"source": "unknown", "num_items": int(len(ds)), "hash": h}


@torch.no_grad()
def _eval_sample_metrics(
    model: torch.nn.Module,
    cfg: dict,
    dev: torch.device,
    timesteps: int,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    eval_steps: int,
    eval_seed: int,
) -> dict[str, float]:
    torch.manual_seed(int(eval_seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(eval_seed))
    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    t_frames = int((cfg["data"]["sample_rate"] * cfg["data"]["clip_seconds"]) / cfg["data"]["stft"]["hop"]) + 1
    xt_mag = torch.randn(1, freq_bins, t_frames, device=dev)
    theta = 2.0 * torch.pi * torch.rand(1, freq_bins, t_frames, device=dev) - torch.pi
    xt_z = phasor_normalize(torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1))

    n = max(1, min(int(eval_steps), int(timesteps)))
    idxs = torch.linspace(timesteps - 1, 0, n, device=dev).round().long().tolist()
    for k, ti in enumerate(idxs):
        t = torch.full((1,), int(ti), device=dev, dtype=torch.long)
        pred_x0_mag, pred_x0_z, _, _ = model(xt_mag, xt_z, t, tokens=None)
        if k + 1 < len(idxs):
            prev = int(idxs[k + 1])
            ab_prev = alphas_cumprod[prev]
            ab_t = alphas_cumprod[int(ti)]
            sqrt_ab_t = ab_t.sqrt()
            sqrt_omab_t = (1.0 - ab_t).sqrt().clamp_min(1e-8)
            sqrt_ab_prev = ab_prev.sqrt()
            sqrt_omab_prev = (1.0 - ab_prev).sqrt()

            eps_mag = (xt_mag - sqrt_ab_t * pred_x0_mag) / sqrt_omab_t
            xt_mag = sqrt_ab_prev * pred_x0_mag + sqrt_omab_prev * eps_mag

            eps_z = (xt_z - sqrt_ab_t * pred_x0_z) / sqrt_omab_t
            xt_z = phasor_normalize(sqrt_ab_prev * pred_x0_z + sqrt_omab_prev * eps_z)
        else:
            xt_mag, xt_z = pred_x0_mag, pred_x0_z

    mag = torch.expm1(xt_mag).clamp_min(0.0)
    phase = phasor_to_phase(xt_z)
    z = mag * torch.exp(1j * phase)
    wav = torch.istft(
        z.squeeze(0),
        n_fft=int(cfg["data"]["stft"]["n_fft"]),
        hop_length=int(cfg["data"]["stft"]["hop"]),
        win_length=int(cfg["data"]["stft"]["win_length"]),
        window=torch.hann_window(int(cfg["data"]["stft"]["win_length"]), device=dev),
        length=int(cfg["data"]["sample_rate"] * cfg["data"]["clip_seconds"]),
    )
    rms = float(torch.sqrt(torch.mean(wav**2) + 1e-12).item())
    peak = float(torch.max(torch.abs(wav)).item())
    top_ratio = float((mag.max(dim=1).values.mean() / mag.mean().clamp_min(1e-8)).item())
    cent_t = spectral_centroid(mag, int(cfg["data"]["sample_rate"]), int(cfg["data"]["stft"]["n_fft"]))
    cent_var = float(cent_t.var(unbiased=False).item())
    return {"rms": rms, "peak": peak, "top_ratio": top_ratio, "cent_var": cent_var}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--model_type", type=str, default=None, choices=["rafa", "baseline"])
    ap.add_argument("--no_clap", action="store_true")
    ap.add_argument("--no_clip", action="store_true")
    ap.add_argument("--ablate_ramanujan", action="store_true")
    ap.add_argument("--ablate_slow_clock", action="store_true")
    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--require_clap", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    dcfg = cfg["diffusion"]
    pcfg = cfg.get("performance", {})
    run_seed = int(dcfg.get("seed", 1337))
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deterministic = bool(pcfg.get("deterministic", False))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
        # Let cuBLAS/cuDNN use Tensor Core friendly paths where possible.
        torch.backends.cuda.matmul.allow_tf32 = bool(pcfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(pcfg.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = False if deterministic else bool(pcfg.get("cudnn_benchmark", True))
        torch.backends.cudnn.deterministic = deterministic
        torch.set_float32_matmul_precision(str(pcfg.get("matmul_precision", "high")))

    base_ds = RafaGuerrillaDataset(cfg)
    genre_names = getattr(base_ds, "GENRES", [])
    dataset_manifest = _dataset_fingerprint(base_ds)
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": run_seed,
        "git_sha": _git_sha(),
        "config_hash": _cfg_hash(cfg),
        "dataset": dataset_manifest,
    }
    print(
        f"run_manifest seed={run_manifest['seed']} git_sha={run_manifest['git_sha']} "
        f"config_hash={run_manifest['config_hash'][:12]} dataset_hash={dataset_manifest['hash'][:12]}"
    )
    use_preload = bool(dcfg.get("preload_dataset", True))
    val_ratio = float(dcfg.get("val_ratio", 0.1))
    ds_train: torch.utils.data.Dataset = base_ds
    ds_val: torch.utils.data.Dataset | None = None
    split_mode = "none"
    if bool(getattr(base_ds, "use_local_files", False)):
        total = len(base_ds)
        if total >= 2 and val_ratio > 0.0:
            val_count = min(total - 1, max(1, int(round(total * val_ratio))))
            idx = list(range(total))
            rng = random.Random(run_seed)
            rng.shuffle(idx)
            val_idx = idx[:val_count]
            train_idx = idx[val_count:]
            if use_preload:
                all_ds = _preload_dataset(base_ds)
                ds_train = Subset(all_ds, train_idx)
                ds_val = Subset(all_ds, val_idx)
            else:
                ds_train = Subset(base_ds, train_idx)
                ds_val = Subset(base_ds, val_idx)
            split_mode = "local_seeded_split"
        elif use_preload:
            ds_train = _preload_dataset(base_ds)
            split_mode = "local_no_val"
    else:
        val_split_name = str(cfg.get("data", {}).get("val_split", "validation"))
        val_raw = RafaGuerrillaDataset(cfg, split=val_split_name)
        if use_preload:
            ds_train = _preload_dataset(base_ds)
            ds_val = _preload_dataset(val_raw)
        else:
            ds_train = base_ds
            ds_val = val_raw
        split_mode = f"gtzan_{val_split_name}"

    run_manifest["data_split"] = {
        "mode": split_mode,
        "train_size": int(len(ds_train)),
        "val_size": int(len(ds_val)) if ds_val is not None else 0,
        "val_ratio": float(val_ratio),
    }
    nw = int(pcfg.get("num_workers", 0))
    loader = DataLoader(
        ds_train,
        batch_size=int(dcfg.get("train_batch_size", 2)),
        shuffle=True,
        num_workers=max(0, nw),
        pin_memory=bool(pcfg.get("pin_memory", False)),
        persistent_workers=bool(pcfg.get("persistent_workers", False)) and nw > 0,
    )
    val_loader = None
    if ds_val is not None and len(ds_val) > 0:
        val_loader = DataLoader(
            ds_val,
            batch_size=int(dcfg.get("train_batch_size", 2)),
            shuffle=False,
            num_workers=max(0, nw),
            pin_memory=bool(pcfg.get("pin_memory", False)),
            persistent_workers=bool(pcfg.get("persistent_workers", False)) and nw > 0,
        )

    model_cfg = cfg
    if bool(dcfg.get("ablation_disable_ramanujan", False)) or args.ablate_ramanujan:
        model_cfg = dict(cfg)
        model_cfg["phase_native_ifs"] = dict(cfg.get("phase_native_ifs", {}))
        model_cfg["phase_native_ifs"]["qset"] = [1]
        model_cfg["phase_native_ifs"]["q_weights"] = [1.0]
    if bool(dcfg.get("ablation_disable_slow_clock", False)) or args.ablate_slow_clock:
        model_cfg = dict(model_cfg)
        model_cfg["phase_native_ifs"] = dict(model_cfg.get("phase_native_ifs", {}))
        model_cfg["phase_native_ifs"]["slow_clock_enabled"] = False
    if not bool(dcfg.get("use_slow_clock", True)):
        model_cfg = dict(model_cfg)
        model_cfg["phase_native_ifs"] = dict(model_cfg.get("phase_native_ifs", {}))
        model_cfg["phase_native_ifs"]["slow_clock_enabled"] = False

    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    model_type = args.model_type if args.model_type is not None else dcfg.get("model_type", "rafa")
    if model_type == "baseline":
        model = BaselineDenoiser(freq_bins).to(dev)
    else:
        model = RAFADenoiser(model_cfg).to(dev)

    tokenizer = None
    use_clap_cond = bool(dcfg.get("use_clap_cond", dcfg.get("use_clip_cond", False))) and (not args.no_clap) and (not args.no_clip)
    require_clap = bool(dcfg.get("require_clap_cond", False)) or bool(args.require_clap)
    use_cfg = bool(dcfg.get("use_cfg", False))
    p_uncond = float(dcfg.get("p_uncond", 0.15))
    if model_type == "baseline" and use_clap_cond:
        print("WARNING: diffusion.model_type=baseline ignores conditioning tokens; disabling use_clap_cond for this run.")
        use_clap_cond = False
    if use_clap_cond:
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
        except Exception as e:
            if require_clap:
                raise RuntimeError(f"CLAP tokenizer load failed for {cfg['model']['text_encoder']}: {e}")
            tokenizer = None
            use_clap_cond = False
            print(f"WARNING: CLAP tokenizer unavailable, falling back to unconditioned training. reason={e}")
    elif require_clap:
        raise RuntimeError("require_clap_cond=true but use_clap_cond=false due config/CLI flags.")

    if bool(pcfg.get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    use_fused_adamw = bool(pcfg.get("fused_adamw", True)) and dev.type == "cuda"
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=float(dcfg.get("lr", 2e-4)), fused=use_fused_adamw)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=float(dcfg.get("lr", 2e-4)))
    timesteps = int(dcfg.get("timesteps", 50))
    betas = make_beta_schedule(
        timesteps,
        float(dcfg.get("beta_start", 1e-4)),
        float(dcfg.get("beta_end", 0.02)),
        dev,
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    scaler = torch.amp.GradScaler("cuda", enabled=bool(pcfg.get("use_amp", True)) and dev.type == "cuda")
    accum = max(1, int(pcfg.get("grad_accum_steps", 1)))
    grad_clip = float(cfg.get("training", {}).get("grad_clip", 0.0) or 0.0)
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else dcfg.get("checkpoint_dir", "checkpoints_diffusion")
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    epochs = int(args.epochs if args.epochs is not None else dcfg.get("train_epochs", 10))
    max_steps = args.max_steps
    eval_every = int(dcfg.get("eval_every_steps", 0))
    eval_steps = int(dcfg.get("eval_sample_steps", 32))
    eval_seeds = dcfg.get("eval_seeds", [dcfg.get("eval_seed", 17)])
    if not isinstance(eval_seeds, (list, tuple)):
        eval_seeds = [eval_seeds]
    eval_seeds = [int(x) for x in eval_seeds]
    eval_rms_min = float(dcfg.get("eval_rms_min", 5e-4))
    eval_cent_var_min = float(dcfg.get("eval_centroid_var_min", 1.0))
    eval_top_ratio_max = float(dcfg.get("eval_top_ratio_max", 30.0))
    eval_fail_on_collapse = bool(dcfg.get("eval_fail_on_collapse", False))
    use_snr_weight = bool(dcfg.get("use_snr_weight", True))
    snr_gamma = float(dcfg.get("snr_gamma", 5.0))
    checkpoint_every = max(1, int(dcfg.get("checkpoint_every_epochs", 1)))
    ckpt_keep_last = dcfg.get("ckpt_keep_last", None)
    ckpt_keep_last = None if ckpt_keep_last is None else max(1, int(ckpt_keep_last))
    val_steps_limit = max(0, int(cfg.get("training", {}).get("val_steps", 0)))
    print(
        f"runtime_knobs model_type={model_type} use_clap_cond={use_clap_cond} grad_clip={grad_clip} "
        f"accum={accum} timesteps={timesteps} deterministic={deterministic} "
        f"train_size={len(ds_train)} val_size={(len(ds_val) if ds_val is not None else 0)}"
    )
    print(f"diffusion_train start model_type={model_type} epochs={epochs} use_clap_cond={use_clap_cond}")
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        n = 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        pending_accum = 0
        for i, (mag, phase, _lbl) in enumerate(loader, start=1):
            mag = mag.to(dev, non_blocking=True)
            phase = phase.to(dev, non_blocking=True)
            x0_mag = torch.log1p(mag)
            x0_z = phase_to_phasor(phase)
            t = torch.randint(0, timesteps, (mag.size(0),), device=dev)
            xt_mag, xt_z, eps_mag, eps_z = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
            tokens = None
            if tokenizer is not None:
                labels = _lbl.detach().cpu().tolist()
                prompts = []
                for li in labels:
                    if 0 <= int(li) < len(genre_names):
                        prompts.append(f"{genre_names[int(li)]} style audio")
                    else:
                        prompts.append("abstract rhythmic audio")
                if use_cfg:
                    mask = torch.rand(len(prompts))
                    prompts = [("" if mask[j].item() < p_uncond else prompts[j]) for j in range(len(prompts))]
                tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                tokens = {k: v.to(dev) for k, v in tokens.items()}

            amp_dtype_name = str(pcfg.get("amp_dtype", "float16")).lower()
            amp_dtype = torch.bfloat16 if amp_dtype_name in ("bfloat16", "bf16") else torch.float16
            with torch.autocast(
                device_type=dev.type,
                dtype=amp_dtype if dev.type == "cuda" else torch.float32,
                enabled=bool(pcfg.get("use_amp", True)) and dev.type == "cuda",
            ):
                pred_mag, pred_z, p_ext, m_ext = model(xt_mag, xt_z, t, tokens=tokens)
                ab_t = extract(alphas_cumprod, t, x0_mag.shape)
                sqrt_ab_t = ab_t.sqrt()
                sqrt_omab_t = (1.0 - ab_t).sqrt().clamp_min(1e-8)

                # Epsilon-pred objective, derived from predicted x0 outputs.
                eps_pred_mag = (xt_mag - sqrt_ab_t * pred_mag) / sqrt_omab_t
                eps_pred_z = (xt_z - sqrt_ab_t.unsqueeze(-1) * pred_z) / sqrt_omab_t.unsqueeze(-1)

                loss_mag_batch = (eps_pred_mag - eps_mag).pow(2).mean(dim=(1, 2))
                loss_phase_batch = (eps_pred_z - eps_z).pow(2).mean(dim=(1, 2, 3))

                if use_snr_weight:
                    snr = (ab_t / (1.0 - ab_t).clamp_min(1e-8)).view(-1)
                    w = torch.minimum(snr, snr.new_full(snr.shape, snr_gamma)) / snr.clamp_min(1e-8)
                    loss_mag = (w * loss_mag_batch).mean()
                    loss_phase = (w * loss_phase_batch).mean()
                else:
                    loss_mag = loss_mag_batch.mean()
                    loss_phase = loss_phase_batch.mean()

                # --- RAFA-PC Physics Losses ---
                l_physics = loss.new_zeros(())
                tw = cfg.get("training", {}).get("weights", {})
                lc = cfg.get("training", {}).get("losses", {})
                rat_ratios = [tuple(x) for x in cfg.get("training", {}).get("rational_ratios", [])]

                # 1. Soft Rational Prior (Milestone 3.0: Learned Frequencies)
                if lc.get("use_rat_prior", True) and "learned_freqs" in p_ext and p_ext["learned_freqs"] is not None:
                    l_rat = rmt.soft_rational_prior(p_ext["learned_freqs"], rat_ratios)
                    l_physics = l_physics + tw.get("w_rat", 0.05) * l_rat

                # 2. Crystal / Harmonic Coupling
                if "phase_state" in p_ext:
                    # [B, T, F, 2] -> (B, F, T) then projected for loss
                    # We take a sample of bins to keep it computationally sane
                    zf = p_ext["phase_state"]
                    phi_f = torch.atan2(zf[..., 1], zf[..., 0]).permute(0, 2, 1) # [B, F, T]
                    freq_major = phi_f.mean(dim=0) # [F, T]
                    
                    if lc.get("use_harmonic_coupling", True):
                        l_harm = rmt.harmonic_coupling_loss(freq_major, rat_ratios)
                        l_physics = l_physics + tw.get("w_harm", 0.05) * l_harm
                    
                    if lc.get("use_crystal_rel", True):
                        l_cry = rmt.ramanujan_crystal_loss(freq_major, qs=tuple(cfg.get("training", {}).get("crystal_qs", [2,3,4,5,6,8,12])))
                        l_physics = l_physics + tw.get("w_crystal", 0.05) * l_cry

                # 3. Holomorphic Consistency (Idea 1: Cauchy-Riemann)
                # The diffusion model should predict a coherent analytic signal.
                # We check the predicted x0 state (pred_mag, pred_phase_from_z)
                pred_phase = phasor_to_phase(pred_z)
                l_holo = rmt.cauchy_riemann_stft_loss(pred_mag.exp(), pred_phase) # mag was log-domain
                l_physics = l_physics + 0.1 * l_holo

                # 4. Crystal-Field Prediction (Idea 5: Denoising Symmetry)
                if "pred_coherence" in p_ext:
                    # Target: Calculate the actual Ramanujan coherence of the clean x0 (approx)
                    # For efficiency, we use a simple spectral entropy or the learned Crystal score
                    # Here we use the actual rhythm strength of the clean target as the ground truth
                    # to teach the model to "see" the rhythm through the noise.
                    with torch.no_grad():
                        # We approximate the "true" coherence from the clean mag/phase target
                        # For now, just use the spectral flux/entropy as a proxy for "interesting structure"
                        target_coh = (x0_mag.exp() - 1.0).mean(dim=2) # [B, F]
                    
                    l_sym = F.mse_loss(p_ext["pred_coherence"], target_coh)
                    l_physics = l_physics + 0.05 * l_sym

                loss = loss_mag + loss_phase + l_physics

            scaler.scale(loss / accum).backward()
            pending_accum += 1
            if i % accum == 0:
                scaler.unscale_(opt)
                if grad_clip > 0.0:
                    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
                else:
                    gnorm = _grad_norm(model)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                pending_accum = 0
            else:
                gnorm = float("nan")
            run += float(loss.detach().item())
            n += 1
            global_step += 1
            if global_step % int(dcfg.get("log_every", 20)) == 0:
                print(f"step={global_step} loss={run / max(1,n):.5f} mag={float(loss_mag.item()):.5f} phase={float(loss_phase.item()):.5f} gnorm={gnorm:.4f}")
            if eval_every > 0 and global_step % eval_every == 0:
                model.eval()
                collapse_any = False
                for s in eval_seeds:
                    m = _eval_sample_metrics(model, cfg, dev, timesteps, betas, alphas_cumprod, eval_steps, int(s))
                    collapse = (m["rms"] < eval_rms_min) or (m["cent_var"] < eval_cent_var_min) or (m["top_ratio"] > eval_top_ratio_max)
                    collapse_any = collapse_any or collapse
                    print(
                        f"eval step={global_step} seed={int(s)} rms={m['rms']:.6f} peak={m['peak']:.6f} "
                        f"top_ratio={m['top_ratio']:.3f} cent_var={m['cent_var']:.3f} collapse={int(collapse)}"
                    )
                model.train()
                if eval_fail_on_collapse and collapse_any:
                    raise RuntimeError(f"Collapse detected at step={global_step} under eval seeds={eval_seeds}.")
            if max_steps is not None and global_step >= int(max_steps):
                break

        if pending_accum > 0:
            scaler.unscale_(opt)
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        avg = run / max(1, n)
        val_avg = float("nan")
        if val_loader is not None:
            model.eval()
            vrun = 0.0
            vn = 0
            with torch.no_grad():
                for vi, (mag, phase, _lbl) in enumerate(val_loader, start=1):
                    mag = mag.to(dev, non_blocking=True)
                    phase = phase.to(dev, non_blocking=True)
                    x0_mag = torch.log1p(mag)
                    x0_z = phase_to_phasor(phase)
                    t = torch.randint(0, timesteps, (mag.size(0),), device=dev)
                    xt_mag, xt_z, eps_mag, eps_z = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
                    tokens = None
                    if tokenizer is not None:
                        labels = _lbl.detach().cpu().tolist()
                        prompts = []
                        for li in labels:
                            if 0 <= int(li) < len(genre_names):
                                prompts.append(f"{genre_names[int(li)]} style audio")
                            else:
                                prompts.append("abstract rhythmic audio")
                        tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                        tokens = {k: v.to(dev) for k, v in tokens.items()}
                    amp_dtype_name = str(pcfg.get("amp_dtype", "float16")).lower()
                    amp_dtype = torch.bfloat16 if amp_dtype_name in ("bfloat16", "bf16") else torch.float16
                    with torch.autocast(
                        device_type=dev.type,
                        dtype=amp_dtype if dev.type == "cuda" else torch.float32,
                        enabled=bool(pcfg.get("use_amp", True)) and dev.type == "cuda",
                    ):
                        pred_mag, pred_z, p_ext, m_ext = model(xt_mag, xt_z, t, tokens=tokens)
                        ab_t = extract(alphas_cumprod, t, x0_mag.shape)
                        sqrt_ab_t = ab_t.sqrt()
                        sqrt_omab_t = (1.0 - ab_t).sqrt().clamp_min(1e-8)
                        eps_pred_mag = (xt_mag - sqrt_ab_t * pred_mag) / sqrt_omab_t
                        eps_pred_z = (xt_z - sqrt_ab_t.unsqueeze(-1) * pred_z) / sqrt_omab_t.unsqueeze(-1)
                        loss_mag_batch = (eps_pred_mag - eps_mag).pow(2).mean(dim=(1, 2))
                        loss_phase_batch = (eps_pred_z - eps_z).pow(2).mean(dim=(1, 2, 3))
                        if use_snr_weight:
                            snr = (ab_t / (1.0 - ab_t).clamp_min(1e-8)).view(-1)
                            w = torch.minimum(snr, snr.new_full(snr.shape, snr_gamma)) / snr.clamp_min(1e-8)
                            loss_base = (w * loss_mag_batch).mean() + (w * loss_phase_batch).mean()
                        else:
                            loss_base = loss_mag_batch.mean() + loss_phase_batch.mean()

                        # Validation Physics
                        l_physics = loss_base.new_zeros(())
                        tw = cfg.get("training", {}).get("weights", {})
                        lc = cfg.get("training", {}).get("losses", {})
                        if lc.get("use_rat_prior", True) and "learned_freqs" in p_ext and p_ext["learned_freqs"] is not None:
                            rat_ratios = [tuple(x) for x in cfg.get("training", {}).get("rational_ratios", [])]
                            l_rat = rmt.soft_rational_prior(p_ext["learned_freqs"], rat_ratios)
                            l_physics = l_physics + tw.get("w_rat", 0.05) * l_rat

                        loss = loss_base + l_physics
                    vrun += float(loss.item())
                    vn += 1
                    if val_steps_limit > 0 and vi >= val_steps_limit:
                        break
            model.train()
            val_avg = vrun / max(1, vn)
        print(f"epoch={ep:03d} train_loss={avg:.5f} val_loss={val_avg:.5f} sec={time.time()-t0:.1f}")
        reached_max = (max_steps is not None and global_step >= int(max_steps))
        should_save = ((ep % checkpoint_every) == 0) or reached_max
        if should_save:
            free_b = shutil.disk_usage(ckpt_dir).free
            if free_b < (512 * 1024 * 1024):
                print(f"WARNING: low disk free space before checkpoint save: {free_b} bytes")
            ckpt_path = os.path.join(ckpt_dir, f"diff_ep{ep}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "run_manifest": run_manifest,
                    "diffusion": {
                        "betas": betas.detach().cpu(),
                        "alphas_cumprod": alphas_cumprod.detach().cpu(),
                        "timesteps": timesteps,
                        "model_type": model_type,
                    },
                },
                ckpt_path,
            )
            try:
                with open(ckpt_path + ".run_manifest.json", "w", encoding="utf-8") as f:
                    json.dump(run_manifest, f, indent=2, sort_keys=True)
            except OSError as e:
                print(f"WARNING: failed to write run manifest sidecar for {ckpt_path}: {e}")
            if ckpt_keep_last is not None:
                _prune_checkpoints(ckpt_dir, ckpt_keep_last)
        if max_steps is not None and global_step >= int(max_steps):
            break


if __name__ == "__main__":
    main()
