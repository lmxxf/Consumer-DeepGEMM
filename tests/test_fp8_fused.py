"""Test FP8 tensor core fused kernel vs float32 reference."""
import torch
import sys
sys.path.insert(0, "/work/Consumer-DeepGEMM")

from consumer_deep_gemm.triton_moe import triton_fused_fp4_matmul_nt


def dequant_ref(b_packed, b_scale_u8, K):
    """Reference dequant: FP4 E2M1 + E8M0 scale -> float32."""
    N, K_packed = b_packed.shape
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        device=b_packed.device, dtype=torch.float32)
    low_f = vals[(b_packed & 0x0F).long()]
    high_f = vals[((b_packed >> 4) & 0x0F).long()]
    out = torch.zeros(N, K, device=b_packed.device, dtype=torch.float32)
    out[:, 0::2] = low_f
    out[:, 1::2] = high_f
    scale_f = torch.pow(2.0, b_scale_u8.float() - 127.0)
    scale_expanded = scale_f.repeat_interleave(32, dim=1)[:, :K]
    return out * scale_expanded


def test_kernel(M, K, N, use_fp8, label):
    K_PACKED = K // 2
    K_SCALE = K // 32

    torch.manual_seed(42)
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_packed = torch.randint(0, 16, (N, K_PACKED), device='cuda', dtype=torch.uint8)
    # Realistic scale range matching V4 weights (exponent 119-122)
    b_scale = torch.randint(119, 123, (N, K_SCALE), device='cuda', dtype=torch.uint8)

    b_deq = dequant_ref(b_packed, b_scale, K)
    ref = a.float() @ b_deq.t()

    out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    triton_fused_fp4_matmul_nt(a, b_packed, b_scale, out, use_fp8=use_fp8)

    diff = (out.float() - ref).abs().max().item()
    rel = diff / (ref.abs().max().item() + 1e-12)
    status = "PASS ✅" if rel < 0.05 else "FAIL ❌"
    print(f"  {label} [{M}x{K}x{N}]: max_abs={diff:.4f} rel_err={rel:.4f} {status}")
    return rel < 0.05


def test_speed(M, K, N):
    K_PACKED = K // 2
    K_SCALE = K // 32
    torch.manual_seed(42)
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_packed = torch.randint(0, 16, (N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_scale = torch.randint(119, 123, (N, K_SCALE), device='cuda', dtype=torch.uint8)

    out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        triton_fused_fp4_matmul_nt(a, b_packed, b_scale, out, use_fp8=True)
        triton_fused_fp4_matmul_nt(a, b_packed, b_scale, out, use_fp8=False)
    torch.cuda.synchronize()

    import time
    N_ITER = 20

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        triton_fused_fp4_matmul_nt(a, b_packed, b_scale, out, use_fp8=False)
    torch.cuda.synchronize()
    t_f32 = (time.perf_counter() - t0) / N_ITER * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        triton_fused_fp4_matmul_nt(a, b_packed, b_scale, out, use_fp8=True)
    torch.cuda.synchronize()
    t_fp8 = (time.perf_counter() - t0) / N_ITER * 1000

    speedup = t_f32 / t_fp8 if t_fp8 > 0 else 0
    print(f"  [{M}x{K}x{N}]: float32={t_f32:.2f}ms  fp8={t_fp8:.2f}ms  speedup={speedup:.2f}x")


if __name__ == "__main__":
    print("=== Correctness (FP8 vs reference) ===")
    all_ok = True
    for M, K, N in [(16, 128, 32), (64, 256, 64), (384, 7168, 4096), (384, 2048, 7168)]:
        ok = test_kernel(M, K, N, use_fp8=True, label="FP8")
        all_ok = all_ok and ok

    print("\n=== Correctness (float32 vs reference) ===")
    for M, K, N in [(16, 128, 32), (384, 7168, 4096)]:
        test_kernel(M, K, N, use_fp8=False, label="F32")

    print("\n=== Speed comparison ===")
    for M, K, N in [(384, 7168, 4096), (384, 2048, 7168)]:
        test_speed(M, K, N)

    print(f"\n{'🎉 All correctness tests passed!' if all_ok else '⚠️ Some tests failed.'}")
