import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import consumer_deep_gemm as dg
from consumer_deep_gemm.gemm import _m_grouped_fp8_fp4_fallback_nt


def test_single_group_mixed_scale():
    M, K, N, G = 128, 256, 128, 1
    a = torch.ones((M, K), dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
    b = torch.full((G, N, K // 2), 0x22, dtype=torch.uint8).view(torch.int8).cuda()
    a_s = torch.ones((M, K // 128), dtype=torch.float32).cuda()
    a_s[:, 0] = 2.0
    a_s[:, 1] = 4.0
    b_s = torch.ones((G, N, K // 32), dtype=torch.float32).cuda()
    d_nat = torch.zeros((M, N), dtype=torch.bfloat16).cuda()
    d_ref = torch.zeros((M, N), dtype=torch.bfloat16).cuda()
    m_idx = torch.zeros(M, dtype=torch.int32).cuda()
    dg.m_grouped_fp8_fp4_gemm_nt_contiguous((a, a_s), (b, b_s), d_nat, m_idx)
    _m_grouped_fp8_fp4_fallback_nt((a, a_s), (b, b_s), d_ref, m_idx)
    torch.cuda.synchronize()
    diff = (d_nat.float() - d_ref.float()).abs().max().item()
    print(f"Test1 G=1 mixed scale: nat={d_nat[0, 0].item():.1f} ref={d_ref[0, 0].item():.1f} diff={diff:.1f}")
    assert diff < 1.0, f"FAIL diff={diff}"


def test_two_groups_large():
    M, K, N, G = 128, 4096, 2048, 2
    a = torch.randn((M, K), dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
    b = torch.randint(0, 256, (G, N, K // 2), dtype=torch.uint8).view(torch.int8).cuda()
    a_s = torch.ones((M, K // 128), dtype=torch.float32).cuda()
    b_s = torch.ones((G, N, K // 32), dtype=torch.float32).cuda()
    m_idx = torch.zeros(M, dtype=torch.int32).cuda()
    m_idx[M // 2:] = 1
    d_nat = torch.zeros((M, N), dtype=torch.bfloat16).cuda()
    d_ref = torch.zeros((M, N), dtype=torch.bfloat16).cuda()
    dg.m_grouped_fp8_fp4_gemm_nt_contiguous((a, a_s), (b, b_s), d_nat, m_idx)
    _m_grouped_fp8_fp4_fallback_nt((a, a_s), (b, b_s), d_ref, m_idx)
    torch.cuda.synchronize()
    nan_count = d_nat.isnan().sum().item()
    valid = ~d_nat.isnan()
    if valid.any():
        diff = (d_nat[valid].float() - d_ref[valid].float()).abs().max().item()
    else:
        diff = float("inf")
    print(f"Test2 G=2 large: nan={nan_count} max_diff={diff:.4f}")
    assert nan_count == 0, f"FAIL nan={nan_count}"
    assert diff < 1.0, f"FAIL diff={diff}"


def test_random_scales():
    M, K, N, G = 128, 256, 128, 1
    a = torch.ones((M, K), dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
    b = torch.full((G, N, K // 2), 0x22, dtype=torch.uint8).view(torch.int8).cuda()
    a_s = torch.tensor([[2.0, 4.0]] * M, dtype=torch.float32).cuda()
    b_s = torch.full((G, N, K // 32), 2.0, dtype=torch.float32).cuda()
    d_nat = torch.zeros((M, N), dtype=torch.bfloat16).cuda()
    d_ref = torch.zeros((M, N), dtype=torch.bfloat16).cuda()
    m_idx = torch.zeros(M, dtype=torch.int32).cuda()
    dg.m_grouped_fp8_fp4_gemm_nt_contiguous((a, a_s), (b, b_s), d_nat, m_idx)
    _m_grouped_fp8_fp4_fallback_nt((a, a_s), (b, b_s), d_ref, m_idx)
    torch.cuda.synchronize()
    diff = (d_nat.float() - d_ref.float()).abs().max().item()
    expected = 128 * 2.0 * 2.0 + 128 * 4.0 * 2.0
    print(f"Test3 both scales: nat={d_nat[0, 0].item():.1f} ref={d_ref[0, 0].item():.1f} expected={expected:.1f} diff={diff:.1f}")
    assert diff < 1.0, f"FAIL diff={diff}"


if __name__ == "__main__":
    test_single_group_mixed_scale()
    test_two_groups_large()
    test_random_scales()
    print("all scale reorder tests passed")
