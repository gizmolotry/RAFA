from __future__ import annotations

import time
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_deq import TRITON_AVAILABLE, _phase_crystal_step_pytorch, dynamic_depth_crystal, phase_crystal_step_triton


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Triton DEQ smoke test.")

    dev = torch.device("cuda")
    torch.manual_seed(1337)

    bsz, q = 4, 129
    z = torch.randn(bsz, q, 2, device=dev, dtype=torch.float32)
    z = z / torch.sqrt((z * z).sum(dim=-1, keepdim=True).clamp_min(1e-8))
    momentum = 0.1 * torch.randn_like(z)
    brain_z = 0.05 * torch.randn_like(z)

    ref_z, ref_delta_sq, _ = _phase_crystal_step_pytorch(z, momentum, brain_z, 0.05)
    tri_z, tri_delta_sq, _ = phase_crystal_step_triton(z, momentum, brain_z, 0.05)

    step_diff = (ref_z - tri_z).abs().max().item()
    delta_diff = (ref_delta_sq - tri_delta_sq).abs().max().item()
    print(f"TRITON_AVAILABLE={int(TRITON_AVAILABLE)}")
    print(f"step_max_diff={step_diff:.8f}")
    print(f"delta_max_diff={delta_diff:.8f}")

    z0 = z.detach().clone().requires_grad_(True)
    brain0 = brain_z.detach().clone().requires_grad_(True)
    out = dynamic_depth_crystal(z0, brain0, momentum.detach().clone(), step_scale=0.05, tol=1e-4, max_steps=50)
    loss = out[..., 0].mean()
    loss.backward()
    print(f"deq_out_shape={tuple(out.shape)}")
    print(f"grad_z0_finite={int(torch.isfinite(z0.grad).all().item())}")
    print(f"grad_brain_finite={int(torch.isfinite(brain0.grad).all().item())}")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(200):
        _ = phase_crystal_step_triton(z, momentum, brain_z, 0.05)
    torch.cuda.synchronize()
    print(f"step_kernel_time_s={(time.time() - t0)/200:.6f}")


if __name__ == "__main__":
    main()
