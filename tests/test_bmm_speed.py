"""Benchmark bmm vs loop mm."""
import torch
import time
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')
from consumer_deep_gemm.gemm import _dequant_fp8_block, _e8m0_to_float
from consumer_deep_gemm.triton_moe import triton_dequant_fp4, _ensure_e8m0_scales

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

    active_groups = m_indices[m_indices >= 0].unique()
    group_rows = []
    group_ids = []
    for gid_t in active_groups:
        gid = gid_t.item()
        rows = (m_indices == gid).nonzero(as_tuple=False).flatten()
        if rows.numel() > 0:
            group_rows.append(rows)
            group_ids.append(gid)
    n_groups = len(group_ids)
    max_rows = max(r.numel() for r in group_rows)

    # Pre-dequant
    b_deq_list = [triton_dequant_fp4(b_packed[gid], b_scale_u8[gid]) for gid in group_ids]
    d_out = torch.zeros(M_sum, N, dtype=torch.bfloat16, device='cuda')

    N_ITER = 20

    # Warmup
    for _ in range(3):
        d_out.zero_()
        for i, (rows, b_deq) in enumerate(zip(group_rows, b_deq_list)):
            a_g = a_deq.index_select(0, rows).to(torch.float32)
            res = torch.mm(a_g, b_deq.to(torch.float32).t())
            d_out.index_copy_(0, rows, res.to(d_out.dtype))

    # === Loop mm ===
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        d_out.zero_()
        for i, (rows, b_deq) in enumerate(zip(group_rows, b_deq_list)):
            a_g = a_deq.index_select(0, rows).to(torch.float32)
            res = torch.mm(a_g, b_deq.to(torch.float32).t())
            d_out.index_copy_(0, rows, res.to(d_out.dtype))
        torch.cuda.synchronize()
    t_loop = (time.perf_counter() - t0) / N_ITER

    # Warmup bmm
    b_batched = torch.stack(b_deq_list, dim=0).to(torch.float32)
    a_batched = torch.zeros(n_groups, max_rows, K, dtype=torch.float32, device='cuda')
    for i, rows in enumerate(group_rows):
        a_batched[i, :rows.numel()] = a_deq.index_select(0, rows).to(torch.float32)
    for _ in range(3):
        torch.bmm(a_batched, b_batched.transpose(1, 2))

    # === BMM ===
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        # Gather A
        for i, rows in enumerate(group_rows):
            a_batched[i, :rows.numel()] = a_deq.index_select(0, rows).to(torch.float32)
        # Single bmm
        result_batched = torch.bmm(a_batched, b_batched.transpose(1, 2))
        # Scatter D
        d_out.zero_()
        for i, rows in enumerate(group_rows):
            d_out.index_copy_(0, rows, result_batched[i, :rows.numel()].to(d_out.dtype))
        torch.cuda.synchronize()
    t_bmm = (time.perf_counter() - t0) / N_ITER

    print(f"{label} [M={M_sum}, K={K}, N={N}]:")
    print(f"  Loop mm:  {t_loop*1000:.2f} ms")
    print(f"  BMM:      {t_bmm*1000:.2f} ms")
    print(f"  Speedup:  {t_loop/t_bmm:.2f}x")
    print()
