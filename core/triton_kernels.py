import torch
import triton
import triton.language as tl

@triton.jit
def _ramanujan_score_logic(
    Z_ptr,  # [B, Q, 2]
    Out_ptr, # [B, Q, Q]
    qset_ptr, # [num_q]
    q_weights_ptr, # [num_q]
    stride_zb, stride_zq,
    stride_ob, stride_oq, stride_oj,
    B, Q, num_q,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    
    rm = pid_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rn = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_m = rm < Q
    mask_n = rn < Q
    
    zi_re = tl.load(Z_ptr + pid_b * stride_zb + rm * stride_zq + 0, mask=mask_m, other=0.0)
    zi_im = tl.load(Z_ptr + pid_b * stride_zb + rm * stride_zq + 1, mask=mask_m, other=0.0)
    
    zj_re = tl.load(Z_ptr + pid_b * stride_zb + rn * stride_zq + 0, mask=mask_n, other=0.0)
    zj_im = tl.load(Z_ptr + pid_b * stride_zb + rn * stride_zq + 1, mask=mask_n, other=0.0)
    
    r_re = zi_re[:, None] * zj_re[None, :] + zi_im[:, None] * zj_im[None, :]
    r_im = zi_im[:, None] * zj_re[None, :] - zi_re[:, None] * zj_im[None, :]
    
    score = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    
    for k in range(num_q):
        q = tl.load(qset_ptr + k)
        w = tl.load(q_weights_ptr + k)
        acc_re = r_re
        acc_im = r_im
        for _ in range(1, q):
            next_re = acc_re * r_re - acc_im * r_im
            next_im = acc_re * r_im + acc_im * r_re
            mag_sq = next_re * next_re + next_im * next_im
            inv_mag = tl.extra.cuda.libdevice.rsqrt(tl.maximum(mag_sq, 1e-12))
            acc_re = next_re * inv_mag
            acc_im = next_im * inv_mag
        score += w * acc_re
        
    tl.store(Out_ptr + pid_b * stride_ob + rm[:, None] * stride_oq + rn[None, :] * stride_oj, score, mask=mask_m[:, None] & mask_n[None, :])

def ramanujan_score_triton(z, qset, q_weights):
    B, Q, _ = z.shape
    out = torch.empty((B, Q, Q), device=z.device, dtype=z.dtype)
    num_q = qset.shape[0]
    BLOCK_SIZE = 16
    grid = (B, triton.cdiv(Q, BLOCK_SIZE), triton.cdiv(Q, BLOCK_SIZE))
    _ramanujan_score_logic[grid](
        z, out, qset, q_weights,
        z.stride(0), z.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        B, Q, num_q,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

@triton.jit
def _fused_ramanujan_summary_logic(
    Z_ptr,  # [B, Q, 2]
    M_ptr,  # [B, Q, 5] - Output Features (re, im, entropy, diag, coherence)
    qset_ptr,
    q_weights_ptr,
    stride_zb, stride_zq,
    stride_mb, stride_mq,
    B, Q, num_q,
    alpha, temp,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    rm = pid_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_m = rm < Q
    
    zi_re = tl.load(Z_ptr + pid_b * stride_zb + rm * stride_zq + 0, mask=mask_m, other=0.0)
    zi_im = tl.load(Z_ptr + pid_b * stride_zb + rm * stride_zq + 1, mask=mask_m, other=0.0)
    
    m_re = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    m_im = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    entropy_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    diag_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    coh_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')
    d_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    COL_BLOCK_SIZE: tl.constexpr = 32
    for j_start in range(0, Q, COL_BLOCK_SIZE):
        rn = j_start + tl.arange(0, COL_BLOCK_SIZE)
        mask_n = rn < Q
        
        zj_re = tl.load(Z_ptr + pid_b * stride_zb + rn * stride_zq + 0, mask=mask_n, other=0.0)
        zj_im = tl.load(Z_ptr + pid_b * stride_zb + rn * stride_zq + 1, mask=mask_n, other=0.0)
        
        r_re = zi_re[:, None] * zj_re[None, :] + zi_im[:, None] * zj_im[None, :]
        r_im = zi_im[:, None] * zj_re[None, :] - zi_re[:, None] * zj_im[None, :]
        
        score = tl.zeros([BLOCK_SIZE, COL_BLOCK_SIZE], dtype=tl.float32)
        for k in range(num_q):
            q = tl.load(qset_ptr + k)
            w = tl.load(q_weights_ptr + k)
            acc_re = r_re
            acc_im = r_im
            for _ in range(1, q):
                next_re = acc_re * r_re - acc_im * r_im
                next_im = acc_re * r_im + acc_im * r_re
                mag_sq = next_re * next_re + next_im * next_im
                inv_mag = tl.extra.cuda.libdevice.rsqrt(tl.maximum(mag_sq, 1e-12))
                acc_re = next_re * inv_mag
                acc_im = next_im * inv_mag
            score += w * acc_re
            
        logits_full = (alpha * score) / temp
        logits = tl.where(mask_n[None, :], logits_full, float('-inf'))
        
        l_next = tl.maximum(l_i, tl.max(logits, 1))
        delta = l_i - l_next
        exp_delta = tl.exp(delta)
        
        p = tl.exp(logits - l_next[:, None])
        d_next = d_i * exp_delta + tl.sum(p, 1)
        
        ent_logits = tl.where(mask_n[None, :], logits - l_next[:, None], 0.0)
        entropy_step = tl.sum(p * ent_logits, 1)
        
        safe_delta_d = tl.where(d_i > 0, delta * d_i, 0.0)
        entropy_acc = exp_delta * (entropy_acc + safe_delta_d) + entropy_step
        
        m_re = m_re * exp_delta + tl.sum(p * r_re, 1)
        m_im = m_im * exp_delta + tl.sum(p * r_im, 1)
        
        coh_acc = coh_acc * exp_delta + tl.sum(p * score, 1)
        
        is_diag = (rm[:, None] == rn[None, :])
        diag_acc = diag_acc * exp_delta + tl.sum(tl.where(is_diag, p, 0.0), 1)
        
        l_i = l_next
        d_i = d_next

    d_i = tl.maximum(d_i, 1e-12)
    m_re /= d_i
    m_im /= d_i
    entropy = tl.log(d_i) - (entropy_acc / d_i)
    diag = diag_acc / d_i
    coherence = coh_acc / d_i
    
    tl.store(M_ptr + pid_b * stride_mb + rm * stride_mq + 0, m_re, mask=mask_m)
    tl.store(M_ptr + pid_b * stride_mb + rm * stride_mq + 1, m_im, mask=mask_m)
    tl.store(M_ptr + pid_b * stride_mb + rm * stride_mq + 2, entropy, mask=mask_m)
    tl.store(M_ptr + pid_b * stride_mb + rm * stride_mq + 3, diag, mask=mask_m)
    tl.store(M_ptr + pid_b * stride_mb + rm * stride_mq + 4, coherence, mask=mask_m)

def ramanujan_summary_triton(z, qset, q_weights, alpha=1.0, temp=1.0):
    B, Q, _ = z.shape
    # Output: [B, Q, 5] (re, im, entropy, diag, coherence)
    m = torch.empty((B, Q, 5), device=z.device, dtype=z.dtype)
    num_q = qset.shape[0]
    BLOCK_SIZE = 16
    grid = (B, triton.cdiv(Q, BLOCK_SIZE))
    _fused_ramanujan_summary_logic[grid](
        z, m, qset, q_weights,
        z.stride(0), z.stride(1),
        m.stride(0), m.stride(1),
        B, Q, num_q,
        alpha=float(alpha), temp=float(temp),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return m
