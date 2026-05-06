"""Test fused dequant+matmul kernel."""
import torch
import time
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')
from consumer_deep_gemm.gemm import _dequant_fp4_block, _dequant_fp8_block, _e8m0_to_float
from consumer_deep_gemm.triton_moe import (
    triton_dequant_fp4, triton_fused_fp4_matmul_nt, _ensure_e8m0_scales,
)


def test_correctness():
    print("=== Fused Matmul Correctness ===")
    torch.manual_seed(42)

    for M, N, K in [(64, 4096, 7168), (1, 4096, 7168), (64, 7168, 2048), (128, 4096, 3584)]:
        K_packed = K // 2

        a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        b_packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device='cuda')
        b_scale = torch.randint(110, 140, (N, K // 32), dtype=torch.uint8, device='cuda')
        b_scale_f32 = _e8m0_to_float(b_scale)

        # Reference: dequant then mm
        b_deq = _dequant_fp4_block(b_packed, b_scale_f32, block_k=32)
        ref = torch.mm(a_bf16.float(), b_deq.float().t()).to(torch.bfloat16)

        # Fused
        out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        triton_fused_fp4_matmul_nt(a_bf16, b_packed, b_scale, out)

        diff = (ref.float() - out.float()).abs()
        max_diff = diff.max().item()
        nz = ref.float().abs() > 1e-6
        rel_err = (diff[nz] / ref.float().abs()[nz]).max().item() if nz.any() else 0.0
        status = "PASS" if rel_err < 0.02 else "FAIL"
        print(f"  [{M:4d}, {N:4d}, {K:4d}]: max_diff={max_diff:.2f}, rel_err={rel_err:.6f} {status}")


def test_speed():
    print("\n=== Fused Matmul Speed ===")
    torch.manual_seed(42)

    for label, M, N, K in [("FC1", 64, 4096, 7168), ("FC2", 64, 7168, 2048)]:
        K_packed = K // 2
        a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        b_packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device='cuda')
        b_scale = torch.randint(110, 140, (N, K // 32), dtype=torch.uint8, device='cuda')
        b_scale_f32 = _e8m0_to_float(b_scale)
        b_scale_u8 = b_scale  # already uint8

        out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

        # Warmup
        for _ in range(5):
            triton_fused_fp4_matmul_nt(a_bf16, b_packed, b_scale_u8, out)
            b_deq = triton_dequant_fp4(b_packed, b_scale_f32)
            torch.mm(a_bf16.float(), b_deq.float().t())

        torch.cuda.synchronize()
        N_ITER = 20

        # Dequant + mm (current)
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            b_deq = triton_dequant_fp4(b_packed, b_scale_f32)
            res = torch.mm(a_bf16.float(), b_deq.float().t())
            torch.cuda.synchronize()
        t_sep = (time.perf_counter() - t0) / N_ITER

        # Fused
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            triton_fused_fp4_matmul_nt(a_bf16, b_packed, b_scale_u8, out)
            torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / N_ITER

        print(f"  {label} [M={M}, N={N}, K={K}]:")
        print(f"    Dequant+mm: {t_sep*1000:.2f} ms")
        print(f"    Fused:      {t_fused*1000:.2f} ms")
        print(f"    Speedup:    {t_sep/t_fused:.2f}x")


if __name__ == '__main__':
    test_correctness()
    test_speed()
    print("\nDone!")
