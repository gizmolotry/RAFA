from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def _renorm_pytorch(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = torch.sqrt((z * z).sum(dim=-1, keepdim=True).clamp_min(eps))
    return z / n


def _phase_crystal_step_pytorch(
    z_current: torch.Tensor,
    previous_momentum: torch.Tensor,
    brain_z: torch.Tensor,
    step_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = z_current + float(step_scale) * previous_momentum + brain_z
    z_next = _renorm_pytorch(pre)
    delta = z_next - z_current
    delta_sq = (delta * delta).sum(dim=-1)
    return z_next, delta_sq, pre


if TRITON_AVAILABLE:
    @triton.jit
    def _phase_crystal_step_kernel(
        z_ptr,
        m_ptr,
        brain_ptr,
        out_ptr,
        delta_ptr,
        n_sites,
        step_scale,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_sites

        base = offs * 2

        z_re = tl.load(z_ptr + base + 0, mask=mask, other=0.0)
        z_im = tl.load(z_ptr + base + 1, mask=mask, other=0.0)
        m_re = tl.load(m_ptr + base + 0, mask=mask, other=0.0)
        m_im = tl.load(m_ptr + base + 1, mask=mask, other=0.0)
        b_re = tl.load(brain_ptr + base + 0, mask=mask, other=0.0)
        b_im = tl.load(brain_ptr + base + 1, mask=mask, other=0.0)

        pre_re = z_re + step_scale * m_re + b_re
        pre_im = z_im + step_scale * m_im + b_im
        norm = tl.sqrt(tl.maximum(pre_re * pre_re + pre_im * pre_im, 1e-8))
        nxt_re = pre_re / norm
        nxt_im = pre_im / norm

        d_re = nxt_re - z_re
        d_im = nxt_im - z_im
        delta_sq = d_re * d_re + d_im * d_im

        tl.store(out_ptr + base + 0, nxt_re, mask=mask)
        tl.store(out_ptr + base + 1, nxt_im, mask=mask)
        tl.store(delta_ptr + offs, delta_sq, mask=mask)


def phase_crystal_step_triton(
    z_current: torch.Tensor,
    previous_momentum: torch.Tensor,
    brain_z: torch.Tensor,
    step_scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        not TRITON_AVAILABLE
        or not z_current.is_cuda
        or not previous_momentum.is_cuda
        or not brain_z.is_cuda
    ):
        return _phase_crystal_step_pytorch(z_current, previous_momentum, brain_z, step_scale)

    if z_current.shape != previous_momentum.shape or z_current.shape != brain_z.shape:
        raise ValueError("z_current, previous_momentum, and brain_z must have identical shapes")
    if z_current.size(-1) != 2:
        raise ValueError("Expected complex phasor layout [..., 2]")

    z_in = z_current.contiguous()
    m_in = previous_momentum.contiguous()
    b_in = brain_z.contiguous()
    n_sites = z_in.numel() // 2
    z_out = torch.empty_like(z_in)
    delta_sq = torch.empty(z_in.shape[:-1], device=z_in.device, dtype=z_in.dtype)

    grid = (triton.cdiv(n_sites, 256),)
    _phase_crystal_step_kernel[grid](
        z_in,
        m_in,
        b_in,
        z_out,
        delta_sq,
        n_sites,
        float(step_scale),
        BLOCK=256,
    )
    pre = z_in + float(step_scale) * m_in + b_in
    return z_out, delta_sq, pre


def _renorm_vjp(pre: torch.Tensor, y: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    # VJP of y = pre / ||pre||.
    norm = torch.sqrt((pre * pre).sum(dim=-1, keepdim=True).clamp_min(1e-8))
    radial = (grad_out * y).sum(dim=-1, keepdim=True)
    return (grad_out - radial * y) / norm


class DynamicDepthCrystalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0: torch.Tensor, momentum0: torch.Tensor, brain_z: torch.Tensor, step_scale: float, tol: float, max_steps: int):
        with torch.no_grad():
            z = z0
            momentum = momentum0
            pre_last = z0 + float(step_scale) * momentum0 + brain_z
            last_delta = torch.tensor(float("inf"), device=z0.device, dtype=z0.dtype)
            steps_taken = 0
            for step_idx in range(int(max_steps)):
                z_next, delta_sq, pre = phase_crystal_step_triton(z, momentum, brain_z, step_scale=float(step_scale))
                last_delta = torch.sqrt(delta_sq.mean().clamp_min(1e-12))
                steps_taken = step_idx + 1
                momentum = z_next - z
                z = z_next
                pre_last = pre
                if float(last_delta.item()) < float(tol):
                    break
        ctx.step_scale = float(step_scale)
        ctx.steps_taken = int(steps_taken)
        ctx.last_delta = float(last_delta.item())
        DynamicDepthCrystalFunction.steps_taken = int(steps_taken)
        DynamicDepthCrystalFunction.last_delta = float(last_delta.item())
        ctx.save_for_backward(pre_last.detach(), z.detach())
        return z

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        pre_last, z_final = ctx.saved_tensors
        grad_pre = _renorm_vjp(pre_last, z_final, grad_out)
        # Single-step Jacobian approximation around the fixed point.
        grad_z0 = grad_pre
        grad_momentum0 = grad_pre * float(ctx.step_scale)
        grad_brain_z = grad_pre
        return grad_z0, grad_momentum0, grad_brain_z, None, None, None


@dataclass
class DynamicDepthCrystalInfo:
    steps_taken: int
    final_delta: float


def dynamic_depth_crystal(
    z0: torch.Tensor,
    brain_z: torch.Tensor,
    previous_momentum: torch.Tensor | None = None,
    *,
    step_scale: float = 0.05,
    tol: float = 1e-4,
    max_steps: int = 50,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, DynamicDepthCrystalInfo]:
    if previous_momentum is None:
        previous_momentum = torch.zeros_like(z0)
    z = DynamicDepthCrystalFunction.apply(
        z0,
        previous_momentum,
        brain_z,
        float(step_scale),
        float(tol),
        int(max_steps),
    )
    if not return_info:
        return z
    return z, DynamicDepthCrystalInfo(
        steps_taken=int(DynamicDepthCrystalFunction.steps_taken if hasattr(DynamicDepthCrystalFunction, "steps_taken") else max_steps),
        final_delta=float(DynamicDepthCrystalFunction.last_delta if hasattr(DynamicDepthCrystalFunction, "last_delta") else math.nan),
    )
