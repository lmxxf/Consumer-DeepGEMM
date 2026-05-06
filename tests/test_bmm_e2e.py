"""End-to-end benchmark: loop (current) vs bmm (proposed)."""
import torch
import time
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')
from consumer_deep_gemm.gemm import _dequant_fp8_block, _e8m0_to_float
from consumer_deep_gemm.triton_moe import (
    triton_dequant_fp4, _ensure_e8m0_scales,
    m_grouped_fp8_fp4_gemm_nt_contiguous_triton,
)

torch.manual_seed(42)

def bmm_version(a_deq, b_tensor, b_scale_u8, m_indices, d, k):
    """BMM-batched grouped GEMM."""
    active_groups = m_indices[m_indices >= 0].unique()
    if active_groups.numel() == 0:
        d.zero_()
        return

    group_rows = []
    group_ids = []
    for gid_t in active_groups:
        gid = gid_t.item()
        rows = (m_indices == gid).nonzero(as_tuple=False).flatten()
        if rows.numel() > 0:
            group_rows.append(rows)
            group_ids.append(gid)

    n_groups = len(group_ids)
    if n_groups == 0:
        d.zero_()
        return
    max_rows = max(r.numel() for r in group_rows)
    N_out = b_tensor.shape[1]

    # Dequant all active experts
    b_deq_list = []
    for gid in group_ids:
        gs = b_scale_u8[gid] if b_scale_u8.dim() == 3 else b_scale_u8
        b_deq_list.append(triton_dequant_fp4(b_tensor[gid], gs))

    # Gather A
    a_batched = torch.zeros(n_groups, max_rows, k, dtype=torch.float32, device=a_deq.device)
    for i, rows in enumerate(group_rows):
        a_batched[i, :rows.numel()] = a_deq.index_select(0, rows).to(torch.float32)

    # Stack B + BMM
    b_batched = torch.stack(b_deq_list, dim=0).to(torch.float32)
    result_batched = torch.bmm(a_batched, b_batched.transpose(1, 2))

    # Scatter
    d.zero_()
    for i, rows in enumerate(group_rows):
        d.index_copy_(0, rows, result_batched[i, :rows.numel()].to(d.dtype))


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
        bmm_version(a_deq, b_packed, b_scale_u8, m_indices, d_out, K)

    # Current loop version
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        m_grouped_fp8_fp4_gemm_nt_contiguous_triton(a_tuple, b_tuple, d_out, m_indices)
        torch.cuda.synchronize()
    t_loop = (time.perf_counter() - t0) / N_ITER

    # BMM version (includes dequant)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        bmm_version(a_deq, b_packed, b_scale_u8, m_indices, d_out, K)
        torch.cuda.synchronize()
    t_bmm = (time.perf_counter() - t0) / N_ITER

    print(f"{label} [M={M_sum}, K={K}, N={N}]:")
    print(f"  Loop (current): {t_loop*1000:.2f} ms")
    print(f"  BMM (proposed): {t_bmm*1000:.2f} ms")
    print(f"  Speedup:        {t_loop/t_bmm:.2f}x")
    print()
