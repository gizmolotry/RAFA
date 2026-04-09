"""Training script for RAFA-PC with actor-specific gradient routing.

Training modes:
- Driver (gru + projection stack): kinematic smoothness and wrapped phase tracking.
- Navigator (attention + clutch): topological alignment and gate regularization.
- Mechanic (scalar suspension params): residual-energy adaptation and frequency-grid constraints.
"""

import math
import os
from typing import Iterable

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import load_config
from dataset import RafaGuerrillaDataset
from model import RAFA
import rafa_math_tools as rmt


def circular_mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return torch.mean(1.0 - torch.cos(pred - tgt))


def compute_delta_phase(phs: torch.Tensor) -> torch.Tensor:
    td = phs[..., 1:] - phs[..., :-1]
    td = ((td + math.pi) % (2.0 * math.pi)) - math.pi
    return torch.cat([torch.zeros_like(td[..., :1]), td], dim=-1)


def _unique_params(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    seen = set()
    out: list[torch.nn.Parameter] = []
    for p in params:
        if p is None:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def build_actor_param_groups(model: RAFA) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    driver: list[torch.nn.Parameter] = []
    navigator: list[torch.nn.Parameter] = []
    mechanic: list[torch.nn.Parameter] = []

    for gear in model.gears:
        driver.extend(list(gear.proj.parameters()))
        driver.extend(list(gear.proj_ph.parameters()))
        driver.extend(list(gear.proj_mag.parameters()))

        if gear.grus is not None:
            for cell in gear.grus:
                driver.extend(list(cell.gru_cell.parameters()))
                navigator.extend(list(cell.attn.parameters()))
        
        if gear.hyena is not None:
            # Hyena is the Navigator's topological engine
            navigator.extend(list(gear.hyena.parameters()))
            if gear.hyena_gate_mlp is not None:
                mechanic.extend(list(gear.hyena_gate_mlp.parameters()))

        mechanic.append(gear.mag_s)

    navigator.extend(list(model.phase_clutch.parameters()))
    navigator.extend(list(model.mag_clutch.parameters()))
    navigator.extend(list(model.phase_fuse.parameters()))
    navigator.extend(list(model.mag_fuse.parameters()))

    if getattr(model, "learned_freqs", None) is not None:
        mechanic.append(model.learned_freqs)

    driver = _unique_params(driver)
    navigator = _unique_params(navigator)
    mechanic = _unique_params(mechanic)

    assigned = {id(p) for p in driver + navigator + mechanic}
    for p in model.parameters():
        if id(p) not in assigned:
            driver.append(p)

    return driver, navigator, mechanic


def _safe_grad(loss: torch.Tensor, params: list[torch.nn.Parameter], retain_graph: bool) -> list[torch.Tensor | None]:
    if not params:
        return []
    return list(torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True))


def _assign_grads(params: list[torch.nn.Parameter], grads: list[torch.Tensor | None]) -> None:
    for p, g in zip(params, grads):
        p.grad = g


def _harmonic_projection(delta_phase: torch.Tensor, bins: int) -> torch.Tensor:
    # delta_phase: (B, F, T) -> (Q, T) expected by harmonic_coupling_loss.
    freq_major = delta_phase.mean(dim=0)
    q = freq_major.shape[0]
    if q <= bins:
        return freq_major
    idx = torch.linspace(0, q - 1, bins, device=delta_phase.device).round().long()
    return freq_major.index_select(0, idx)


def main() -> None:
    cfg = load_config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RafaGuerrillaDataset(cfg)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model = RAFA(cfg).to(dev)

    driver_params, navigator_params, mechanic_params = build_actor_param_groups(model)

    base_lr = float(cfg["training"]["lr"])
    driver_lr = float(cfg["training"].get("driver_lr", base_lr))
    navigator_lr = float(cfg["training"].get("navigator_lr", base_lr))
    mechanic_lr = float(cfg["training"].get("mechanic_lr", 1e-5))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    max_steps = cfg["training"].get("max_steps_per_epoch", None)

    opt_driver = optim.Adam(driver_params, lr=driver_lr)
    opt_navigator = optim.Adam(navigator_params, lr=navigator_lr)
    opt_mechanic = optim.Adam(mechanic_params, lr=mechanic_lr)

    text_cond = cfg["model"].get("text_conditioned", False)
    losses_cfg = cfg["training"]["losses"]
    weights = cfg["training"]["weights"]
    rat_ratios = [tuple(x) for x in cfg["training"]["rational_ratios"]]
    vm_cfg = cfg["training"]["von_mises"]

    use_r2 = losses_cfg.get("use_r2_modular_embed", True)
    use_r5 = losses_cfg.get("use_r5_adaptive_von_mises", True)
    use_mag = losses_cfg.get("use_mag_mse", True)
    use_con = losses_cfg.get("use_contrastive", False) and text_cond
    use_rat = losses_cfg.get("use_rat_prior", True) and getattr(model, "learned_freqs", None) is not None
    use_harm = losses_cfg.get("use_harmonic_coupling", True)
    use_crystal = losses_cfg.get("use_crystal_rel", True)
    use_mono = losses_cfg.get("use_monotonic_freqs", False)
    use_band = losses_cfg.get("use_band_clamp", False)

    band_min = float(cfg["training"].get("band_min", 0.0))
    band_max = float(cfg["training"].get("band_max", math.pi))
    harmonic_bins = int(cfg["training"].get("harmonic_bins", 64))
    crystal_qs = tuple(cfg["training"].get("crystal_qs", [2, 3, 4, 5, 6, 8, 12]))
    crystal_gamma = float(cfg["training"].get("crystal_gamma", 1.0))
    crystal_alpha = float(cfg["training"].get("crystal_alpha", 1.0))
    crystal_temp = float(cfg["training"].get("crystal_temp", 1.0))
    clutch_iso_enabled = bool(cfg["training"].get("clutch_isolation_enabled", True))
    clutch_transient_gear = int(cfg["training"].get("clutch_transient_gear_idx", 0))
    clutch_steady_gear = int(cfg["training"].get("clutch_steady_gear_idx", 1))

    checkpoint_dir = cfg["training"]["checkpoint_dir"]
    checkpoint_every_steps = int(cfg["training"].get("checkpoint_every_steps", 1000))
    os.makedirs(checkpoint_dir, exist_ok=True)

    for ep in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        running = 0.0
        steps = 0
        nan_grad_count = 0
        last_driver_loss = torch.zeros((), device=dev)
        last_navigator_loss = torch.zeros((), device=dev)
        last_mechanic_loss = torch.zeros((), device=dev)

        for mags, phs, labels in loader:
            mags = mags.to(dev)
            phs = phs.to(dev)
            _ = labels

            if not torch.isfinite(mags).all() or not torch.isfinite(phs).all():
                print(f"NaN detected in input batch at step {steps}!")
                continue

            pd, md, phase_extras, mag_extras = model(mags, phs, None)
            td = compute_delta_phase(phs)

            l_phase = circular_mse(pd, td)
            l_mag = F.mse_loss(md, mags) if use_mag else pd.new_zeros(())

            d_pd = compute_delta_phase(pd)
            d_td = compute_delta_phase(td)
            l_kin_smooth = F.mse_loss(d_pd, d_td)

            driver_loss = weights.get("w_r1", 1.0) * l_phase + 0.2 * l_kin_smooth

            l_r2 = rmt.modular_phase_embed_loss(pd, td, qs=(3, 4, 5), reduction="mean") if use_r2 else pd.new_zeros(())
            if use_r5:
                kappa = pd.new_tensor(float(vm_cfg.get("kappa0", 1.0)))
                l_r5 = rmt.von_mises_nll(pd, td, kappa).mean()
            else:
                l_r5 = pd.new_zeros(())

            gate_reg = pd.new_zeros(())
            if isinstance(phase_extras, dict) and "gate_reg_loss" in phase_extras:
                gate_reg = gate_reg + phase_extras["gate_reg_loss"]
            if isinstance(mag_extras, dict) and "gate_reg_loss" in mag_extras:
                gate_reg = gate_reg + mag_extras["gate_reg_loss"]
            ifs_router_reg = pd.new_zeros(())
            if isinstance(phase_extras, dict) and "ifs_debug" in phase_extras:
                dbg = phase_extras["ifs_debug"]
                if isinstance(dbg, dict) and "router_entropy_reg" in dbg:
                    ifs_router_reg = dbg["router_entropy_reg"] if torch.is_tensor(dbg["router_entropy_reg"]) else pd.new_tensor(float(dbg["router_entropy_reg"]))

            l_clutch_iso = pd.new_zeros(())
            l_clutch_balance = pd.new_zeros(())
            if clutch_iso_enabled and isinstance(phase_extras, dict) and "phase_gates" in phase_extras:
                gates = phase_extras["phase_gates"]
                if torch.is_tensor(gates) and gates.dim() == 2 and gates.size(1) > 1:
                    n_gears = gates.size(1)
                    t_idx = max(0, min(clutch_transient_gear, n_gears - 1))
                    s_idx = max(0, min(clutch_steady_gear, n_gears - 1))
                    flux = (mags[..., 1:] - mags[..., :-1]).abs().mean(dim=(1, 2))
                    flux_target = flux / (flux + 1.0)
                    trans_target = flux_target.detach()
                    steady_target = (1.0 - flux_target).detach()
                    l_clutch_iso = (
                        F.mse_loss(gates[:, t_idx], trans_target)
                        + F.mse_loss(gates[:, s_idx], steady_target)
                    )
                    gate_mean = gates.mean(dim=0)
                    uniform = gate_mean.new_full((n_gears,), 1.0 / float(n_gears))
                    l_clutch_balance = F.mse_loss(gate_mean, uniform)

            l_harm = pd.new_zeros(())
            if use_harm:
                harmonic_input = _harmonic_projection(pd, harmonic_bins)
                l_harm = rmt.harmonic_coupling_loss(harmonic_input, rat_ratios, lam=1.0)

            l_crystal = pd.new_zeros(())
            if use_crystal:
                crystal_input = _harmonic_projection(pd, harmonic_bins)
                l_crystal = rmt.ramanujan_crystal_loss(
                    crystal_input,
                    qs=crystal_qs,
                    gamma=crystal_gamma,
                    alpha=crystal_alpha,
                    temp=crystal_temp,
                )

            l_contrast = pd.new_zeros(())
            if use_con:
                # Placeholder: dataset currently does not provide text tokens in active pipeline.
                l_contrast = pd.new_zeros(())

            navigator_loss = (
                weights.get("w_r2", 0.1) * l_r2
                + weights.get("w_r5", 0.1) * l_r5
                + weights.get("w_harm", 0.05) * l_harm
                + weights.get("w_crystal", 0.05) * l_crystal
                + gate_reg
                + ifs_router_reg
                + weights.get("w_clutch_iso", 0.2) * l_clutch_iso
                + weights.get("w_clutch_balance", 0.05) * l_clutch_balance
                + weights.get("w_contrast", 1.0) * l_contrast
            )

            l_rat = pd.new_zeros(())
            l_band = pd.new_zeros(())
            l_mono = pd.new_zeros(())
            if use_rat and getattr(model, "learned_freqs", None) is not None:
                l_rat = rmt.soft_rational_prior(model.learned_freqs, rat_ratios)

            if getattr(model, "learned_freqs", None) is not None and use_band:
                l_band = (F.relu(band_min - model.learned_freqs) + F.relu(model.learned_freqs - band_max)).mean()

            if getattr(model, "learned_freqs", None) is not None and use_mono:
                l_mono = F.relu(-(model.learned_freqs[1:] - model.learned_freqs[:-1])).mean()

            scalar_reg = pd.new_zeros(())
            for gear in model.gears:
                scalar_reg = scalar_reg + (gear.mag_s - 1.0).pow(2)
            scalar_reg = scalar_reg / max(1, len(model.gears))

            mechanic_loss = (
                weights.get("w_mag", 1.0) * l_mag
                + 0.25 * l_phase
                + weights.get("w_rat", 0.05) * l_rat
                + 0.1 * l_band
                + 0.1 * l_mono
                + 0.01 * scalar_reg
            )
            last_driver_loss = driver_loss.detach()
            last_navigator_loss = navigator_loss.detach()
            last_mechanic_loss = mechanic_loss.detach()

            # Compute actor-specific gradients from a shared forward graph.
            driver_grads = _safe_grad(driver_loss, driver_params, retain_graph=True)
            navigator_grads = _safe_grad(navigator_loss, navigator_params, retain_graph=True)
            mechanic_grads = _safe_grad(mechanic_loss, mechanic_params, retain_graph=False)

            def _check_grads(grads, name):
                for i, g in enumerate(grads):
                    if g is not None and not torch.isfinite(g).all():
                        return True
                return False

            if _check_grads(driver_grads, "Driver") or _check_grads(navigator_grads, "Navigator") or _check_grads(mechanic_grads, "Mechanic"):
                nan_grad_count += 1
                if steps % 100 == 0:
                    print(f"[{steps}] NaN gradients detected {nan_grad_count} times so far. Skipping update.")
                continue

            opt_driver.zero_grad(set_to_none=True)
            _assign_grads(driver_params, driver_grads)
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(driver_params, grad_clip)
            opt_driver.step()

            opt_navigator.zero_grad(set_to_none=True)
            _assign_grads(navigator_params, navigator_grads)
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(navigator_params, grad_clip)
            opt_navigator.step()

            opt_mechanic.zero_grad(set_to_none=True)
            _assign_grads(mechanic_params, mechanic_grads)
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(mechanic_params, grad_clip)
            opt_mechanic.step()

            total = driver_loss.detach() + navigator_loss.detach() + mechanic_loss.detach()
            if not torch.isfinite(total):
                print(f"NaN detected at step {steps}!")
                print(f"Driver loss: {driver_loss.item()}")
                print(f"Navigator loss: {navigator_loss.item()}")
                print(f"Mechanic loss: {mechanic_loss.item()}")
                # Check model parameters
                for name, p in model.named_parameters():
                    if not torch.isfinite(p).all():
                        print(f"NaN in parameter: {name}")
                raise RuntimeError("Loss became NaN")

            running += float(total.item())
            steps += 1

            if steps % checkpoint_every_steps == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"rafa_ep{ep}_step{steps}.pt")
                torch.save(model.state_dict(), ckpt_path)
                # Keep a 'latest' symlink or copy for convenience
                latest_path = os.path.join(checkpoint_dir, "rafa_latest.pt")
                torch.save(model.state_dict(), latest_path)
                print(f"Periodic checkpoint saved -> {ckpt_path}")

            if max_steps is not None and steps >= int(max_steps):
                break

        avg = running / max(1, steps)
        print(
            f"Epoch {ep:03d} | loss={avg:.5f} "
            f"| driver={float(last_driver_loss.item()):.5f} "
            f"| nav={float(last_navigator_loss.item()):.5f} "
            f"| mech={float(last_mechanic_loss.item()):.5f} "
            f"| skipped={nan_grad_count}"
        )
        
        # Log to file
        with open("logs/rafa_training.log", "a") as f:
            f.write(f"epoch={ep} steps={steps} loss={avg:.5f} driver={float(last_driver_loss.item()):.5f} nav={float(last_navigator_loss.item()):.5f} mech={float(last_mechanic_loss.item()):.5f} skipped={nan_grad_count}\n")

        ckpt_path = os.path.join(checkpoint_dir, f"rafa_ep{ep}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()

