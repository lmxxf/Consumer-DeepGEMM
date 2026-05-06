"""Test Triton grouped FP4 MoE GEMM against Python reference."""
import torch
import time
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')

from consumer_deep_gemm.gemm import (
    _dequant_fp4_block, _dequant_fp8_block, _e8m0_to_float,
    _m_grouped_fp8_fp4_dequant_mm_nt,
)
from consumer_deep_gemm.triton_moe import m_grouped_fp8_fp4_gemm_nt_contiguous_triton


def make_test_data(M_sum=384, K=7168, N=4096, num_experts=256, top_k=6):
    """Create test data mimicking V4 Flash decode MoE."""
    # FP8 activation + scale
    a_tensor = torch.randn(M_sum, K, device='cuda').to(torch.float8_e4m3fn)
    a_scale = torch.ones(M_sum, K // 128, device='cuda', dtype=torch.float32)

    # FP4 packed weights + E8M0 scales
    b_packed = torch.randint(0, 256, (num_experts, N, K // 2), dtype=torch.uint8, device='cuda')
    b_scale = torch.randint(110, 140, (num_experts, N, K // 32), dtype=torch.uint8, device='cuda')
    b_scale_f32 = _e8m0_to_float(b_scale)

    # m_indices: assign rows to top_k experts (contiguous layout)
    m_indices = torch.full((M_sum,), -1, dtype=torch.int32, device='cuda')
    rows_per_expert = M_sum // top_k
    expert_ids = torch.randperm(num_experts)[:top_k]
    for i, eid in enumerate(expert_ids):
        start = i * rows_per_expert
        end = start + rows_per_expert
        m_indices[start:end] = eid.item()

    d_ref = torch.empty(M_sum, N, dtype=torch.bfloat16, device='cuda')
    d_tri = torch.empty(M_sum, N, dtype=torch.bfloat16, device='cuda')

    return (a_tensor, a_scale), (b_packed, b_scale_f32), m_indices, d_ref, d_tri


def test_correctness():
    print("=== Grouped GEMM Correctness ===")
    a, b, m_indices, d_ref, d_tri = make_test_data()

    _m_grouped_fp8_fp4_dequant_mm_nt(a, b, d_ref, m_indices)
    m_grouped_fp8_fp4_gemm_nt_contiguous_triton(a, b, d_tri, m_indices)

    diff = (d_ref.float() - d_tri.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # Relative error on non-zero elements
    nz = d_ref.float().abs() > 1e-6
    if nz.any():
        rel_err = (diff[nz] / d_ref.float().abs()[nz]).max().item()
    else:
        rel_err = 0.0
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}, max_rel_err={rel_err:.6f}")
    assert max_diff < 1.0 or rel_err < 0.01, f"FAIL: max_diff={max_diff}, rel_err={rel_err}"
    print("  PASS")


def test_speed():
    print("\n=== Grouped GEMM Speed ===")

    # FC1 shape: M_sum=384, K=7168, N=4096
    for label, M_sum, K, N in [
        ("FC1", 384, 7168, 4096),
        ("FC2", 384, 2048, 7168),
    ]:
        a, b, m_indices, d_ref, d_tri = make_test_data(M_sum=M_sum, K=K, N=N)

        # Warmup
        for _ in range(3):
            _m_grouped_fp8_fp4_dequant_mm_nt(a, b, d_ref, m_indices)
            m_grouped_fp8_fp4_gemm_nt_contiguous_triton(a, b, d_tri, m_indices)

        torch.cuda.synchronize()

        # Python
        N_ITER = 10
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _m_grouped_fp8_fp4_dequant_mm_nt(a, b, d_ref, m_indices)
            torch.cuda.synchronize()
        t_py = (time.perf_counter() - t0) / N_ITER

        # Triton
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            m_grouped_fp8_fp4_gemm_nt_contiguous_triton(a, b, d_tri, m_indices)
            torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / N_ITER

        print(f"  {label} [M={M_sum}, K={K}, N={N}]:")
        print(f"    Python:  {t_py*1000:.2f} ms")
        print(f"    Triton:  {t_tri*1000:.2f} ms")
        print(f"    Speedup: {t_py/t_tri:.1f}x")


def test_breakdown():
    """Profile individual steps of the Triton path."""
    print("\n=== Step Breakdown (Triton path) ===")
    a_tuple, b_tuple, m_indices, d_ref, d_tri = make_test_data(M_sum=384, K=7168, N=4096)
    a_tensor, a_scale = a_tuple
    b_tensor, b_scale = b_tuple

    from consumer_deep_gemm.triton_moe import triton_dequant_fp4
    from consumer_deep_gemm.gemm import _dequant_fp8_block

    # Warmup
    for _ in range(3):
        m_grouped_fp8_fp4_gemm_nt_contiguous_triton(a_tuple, b_tuple, d_tri, m_indices)
    torch.cuda.synchronize()

    N_ITER = 10

    # Step 1: FP8 dequant
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        a_deq = _dequant_fp8_block(a_tensor, a_scale)
        torch.cuda.synchronize()
    t_fp8 = (time.perf_counter() - t0) / N_ITER

    # Step 2: unique + nonzero
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        active = m_indices[m_indices >= 0].unique()
        for gid_t in active:
            mask = (m_indices == gid_t.item())
            rows = mask.nonzero(as_tuple=False).flatten()
        torch.cuda.synchronize()
    t_idx = (time.perf_counter() - t0) / N_ITER

    # Step 3: FP4 dequant (all 6 experts)
    active = m_indices[m_indices >= 0].unique()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        for gid_t in active:
            gid = gid_t.item()
            b_group_deq = triton_dequant_fp4(b_tensor[gid], b_scale[gid])
        torch.cuda.synchronize()
    t_fp4 = (time.perf_counter() - t0) / N_ITER

    # Step 4: matmul (all 6 experts)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        d_tri.zero_()
        for gid_t in active:
            gid = gid_t.item()
            mask = (m_indices == gid_t.item())
            rows = mask.nonzero(as_tuple=False).flatten()
            b_group_deq = triton_dequant_fp4(b_tensor[gid], b_scale[gid])
            a_group = a_deq.index_select(0, rows).to(torch.float32)
            result = torch.mm(a_group, b_group_deq.to(torch.float32).t())
            d_tri.index_copy_(0, rows, result.to(d_tri.dtype))
        torch.cuda.synchronize()
    t_full = (time.perf_counter() - t0) / N_ITER

    print(f"  FP8 dequant:     {t_fp8*1000:.2f} ms")
    print(f"  Index ops:       {t_idx*1000:.2f} ms")
    print(f"  FP4 dequant ×6:  {t_fp4*1000:.2f} ms")
    print(f"  Full (dequant+mm): {t_full*1000:.2f} ms")


if __name__ == '__main__':
    test_correctness()
    test_speed()
    test_breakdown()
    print("\nDone!")
