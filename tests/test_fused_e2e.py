"""End-to-end: fused kernel in grouped GEMM vs current Triton dequant + mm."""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from consumer_deep_gemm.gemm import _dequant_fp8_block, _e8m0_to_float
from consumer_deep_gemm.triton_moe import (
    triton_dequant_fp4, triton_fused_fp4_matmul_nt,
    _ensure_e8m0_scales, m_grouped_fp8_fp4_gemm_nt_contiguous_triton,
)


def fused_grouped_gemm(a_deq, b_tensor, b_scale_u8, m_indices, d, k):
    """Grouped GEMM using fused dequant+matmul kernel."""
    active_groups = m_indices[m_indices >= 0].unique()
    if active_groups.numel() == 0:
        d.zero_()
        return

    d.zero_()
    for gid_t in active_groups:
        gid = gid_t.item()
        rows = (m_indices == gid).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue

        gs = b_scale_u8[gid] if b_scale_u8.dim() == 3 else b_scale_u8
        a_group = a_deq.index_select(0, rows)

        # Fused dequant + matmul in one kernel
        result = torch.empty(rows.numel(), b_tensor.shape[1], dtype=torch.bfloat16, device=d.device)
        triton_fused_fp4_matmul_nt(a_group, b_tensor[gid], gs, result)

        d.index_copy_(0, rows, result)


torch.manual_seed(42)

for label, M_sum, K, N in [("FC1", 384, 7168, 4096), ("FC2", 384, 2048, 7168)]:
    num_experts, top_k = 256, 6

    a_tensor = torch.randn(M_sum, K, device='cuda').to(torch.float8_e4m3fn)
    a_scale = torch.ones(M_sum, K // 128, device='cuda', dtype=torch.float32)
    a_deq = _dequant_fp8_block(a_tensor, a_scale)

    b_packed = torch.randint(0, 256, (num_experts, N, K // 2), dtype=torch.uint8, device='cuda')
    b_scale = torch.randint(110, 140, (num_experts, N, K // 32), dtype=torch.uint8, device='cuda')
    b_scale_f32 = _e8m0_to_float(b_scale)
    b_scale_u8 = _ensure_e8m0_scales(b_scale_f32)

    m_indices = torch.full((M_sum,), -1, dtype=torch.int32, device='cuda')
    rows_per = M_sum // top_k
    eids = torch.randperm(num_experts)[:top_k]
    for i, eid in enumerate(eids):
        m_indices[i * rows_per:(i+1) * rows_per] = eid.item()

    a_tuple = (a_tensor, a_scale)
    b_tuple = (b_packed, b_scale_f32)
    d_out = torch.empty(M_sum, N, dtype=torch.bfloat16, device='cuda')

    N_ITER = 20

    # Warmup
    for _ in range(3):
        m_grouped_fp8_fp4_gemm_nt_contiguous_triton(a_tuple, b_tuple, d_out, m_indices)
        fused_grouped_gemm(a_deq, b_packed, b_scale_u8, m_indices, d_out, K)

    # Current default path.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        m_grouped_fp8_fp4_gemm_nt_contiguous_triton(a_tuple, b_tuple, d_out, m_indices)
        torch.cuda.synchronize()
    t_current = (time.perf_counter() - t0) / N_ITER

    # Baseline: explicit one fused dequant+matmul kernel per active expert.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        fused_grouped_gemm(a_deq, b_packed, b_scale_u8, m_indices, d_out, K)
        torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / N_ITER

    print(f"{label} [M={M_sum}, K={K}, N={N}]:")
    print(f"  Current default:      {t_current*1000:.2f} ms")
    print(f"  Explicit baseline:    {t_fused*1000:.2f} ms")
    print(f"  Current/baseline:     {t_fused/t_current:.2f}x")
    print()
