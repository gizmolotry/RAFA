
import torch
import math
from triton_rnns import ramanujan_score_triton

def _mul(a, b):
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)

def _renorm(z, eps=1e-8):
    n = torch.sqrt((z * z).sum(dim=-1, keepdim=True).clamp_min(eps))
    return z / n

def ramanujan_score_pytorch(z, qset, q_weights):
    # z: [B, Q, 2]
    # Build r: [B, Q, Q, 2]
    zi = z.unsqueeze(2)
    zj = z.unsqueeze(1)
    # conj(zj)
    zj_conj = zj.clone()
    zj_conj[..., 1] = -zj_conj[..., 1]
    
    r = _mul(zi, zj_conj)
    
    out = torch.zeros(r.shape[0], r.shape[1], r.shape[2], device=z.device, dtype=z.dtype)
    for q, w in zip(qset, q_weights):
        rq = r
        for _ in range(1, int(q)):
            rq = _mul(rq, r)
            rq = _renorm(rq)
        out = out + float(w) * rq[..., 0]
    return out

def test_rnn():
    B, Q = 2, 129
    qset = torch.tensor([2, 3, 4, 5], device="cuda", dtype=torch.int32)
    q_weights = torch.tensor([1.0, 0.9, 0.8, 0.7], device="cuda", dtype=torch.float32)
    
    z = torch.randn(B, Q, 2, device="cuda")
    z = _renorm(z)
    
    expected = ramanujan_score_pytorch(z, qset, q_weights)
    actual = ramanujan_score_triton(z, qset, q_weights)
    
    diff = (expected - actual).abs().max().item()
    print(f"Max difference: {diff:.6f}")
    if diff < 1e-4:
        print("SUCCESS")
    else:
        print("FAILURE")

    # Benchmark
    import time
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = ramanujan_score_pytorch(z, qset, q_weights)
    torch.cuda.synchronize()
    print(f"PyTorch time: {(time.time()-t0)/100:.6f}s")
    
    t0 = time.time()
    for _ in range(100):
        _ = ramanujan_score_triton(z, qset, q_weights)
    torch.cuda.synchronize()
    print(f"Triton time: {(time.time()-t0)/100:.6f}s")

    # Test Summary rnn
    from triton_rnns import ramanujan_summary_triton
    
    def _relative_summary_pytorch(z, a):
        zi = z.unsqueeze(2)
        zj = z.unsqueeze(1)
        zj_conj = zj.clone()
        zj_conj[..., 1] = -zj_conj[..., 1]
        r = _mul(zi, zj_conj)
        return (a.unsqueeze(-1) * r).sum(dim=2)

    alpha, temp = 0.5, 2.0
    a = torch.softmax((alpha * expected) / temp, dim=-1) # using the scores from before as logits
    expected_summary = _relative_summary_pytorch(z, a)
    actual_summary_full = ramanujan_summary_triton(z, qset, q_weights, alpha=alpha, temp=temp)
    actual_summary = actual_summary_full[..., :2]
    actual_entropy = actual_summary_full[..., 2]
    actual_diag = actual_summary_full[..., 3]
    
    diff_summary = (expected_summary - actual_summary).abs().max().item()
    print(f"Summary (alpha={alpha}, temp={temp}) Max difference: {diff_summary:.6f}")
    
    # Expected Entropy
    expected_entropy = -(a * torch.log(a.clamp_min(1e-12))).sum(dim=-1)
    diff_entropy = (expected_entropy - actual_entropy).abs().max().item()
    print(f"Entropy Max difference: {diff_entropy:.6f}")
    
    # Expected Diag
    expected_diag = torch.diagonal(a, dim1=1, dim2=2)
    diff_diag = (expected_diag - actual_diag).abs().max().item()
    print(f"Diag Max difference: {diff_diag:.6f}")

    if diff_summary < 1e-4 and diff_entropy < 1e-4 and diff_diag < 1e-4:
        print("SUMMARY SUCCESS")
    else:
        print("SUMMARY FAILURE")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = _relative_summary_pytorch(z, a)
    torch.cuda.synchronize()
    print(f"PyTorch Summary time: {(time.time()-t0)/100:.6f}s")
    
    t0 = time.time()
    for _ in range(100):
        _ = ramanujan_summary_triton(z, qset, q_weights)
    torch.cuda.synchronize()
    print(f"Triton Summary time: {(time.time()-t0)/100:.6f}s")

if __name__ == "__main__":
    test_rnn()
