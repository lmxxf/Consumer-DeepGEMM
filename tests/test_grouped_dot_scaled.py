import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consumer_deep_gemm.gemm import _dequant_fp8_block, _e8m0_to_float
from consumer_deep_gemm.triton_moe import (
    _ensure_e8m0_scales,
    m_grouped_fp8_fp4_gemm_nt_contiguous_triton,
    triton_fused_fp4_matmul_nt,
)


def _reference_grouped(a_deq, b_tensor, b_scale_u8, m_indices, out):
    out.zero_()
    for gid_t in m_indices[m_indices >= 0].unique():
        gid = int(gid_t.item())
        rows = (m_indices == gid).nonzero(as_tuple=False).flatten()
        result = torch.empty(rows.numel(), b_tensor.shape[1],
                             dtype=out.dtype, device=out.device)
        triton_fused_fp4_matmul_nt(a_deq.index_select(0, rows),
                                   b_tensor[gid],
                                   b_scale_u8[gid],
                                   result)
        out.index_copy_(0, rows, result)


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping")
        return

    if os.getenv("CDG_RUN_UNSAFE_GROUPED_TEST", "0") != "1":
        print("Grouped dot_scaled test is disabled by default; set CDG_RUN_UNSAFE_GROUPED_TEST=1 to run it.")
        return

    os.environ["CDG_GROUPED_DOT_SCALED"] = "1"
    torch.manual_seed(7)
    m, k, n = 96, 256, 128
    groups = 8

    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    a_scale = torch.ones(m, k // 128, device="cuda", dtype=torch.float32)
    a_deq = _dequant_fp8_block(a, a_scale)

    b = torch.randint(0, 256, (groups, n, k // 2),
                      dtype=torch.uint8, device="cuda")
    b_scale_u8 = torch.randint(120, 132, (groups, n, k // 32),
                               dtype=torch.uint8, device="cuda")
    b_scale_f32 = _e8m0_to_float(b_scale_u8)
    b_scale_u8_cached = _ensure_e8m0_scales(b_scale_f32)

    m_indices = torch.full((m,), -1, dtype=torch.int32, device="cuda")
    for i, gid in enumerate([5, 1, 7, 3, 5, 1]):
        m_indices[i * 12:(i + 1) * 12] = gid

    actual = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    expected = torch.empty_like(actual)

    m_grouped_fp8_fp4_gemm_nt_contiguous_triton(
        (a, a_scale), (b, b_scale_f32), actual, m_indices)
    _reference_grouped(a_deq, b, b_scale_u8_cached, m_indices, expected)

    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs().max().item()
    print(f"max_diff={diff:.6f}")
    assert diff == 0.0


if __name__ == "__main__":
    main()
