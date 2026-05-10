"""Test dot_scaled matmul kernel integrated into triton_moe.py."""
import time
import torch
import sys
sys.path.insert(0, "/work/Consumer-DeepGEMM")

from consumer_deep_gemm.triton_moe import triton_fused_fp4_matmul_nt


def dequant_ref(b_packed, b_scale_u8, K):
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


def test(M, K, N, use_fp8, label):
    torch.manual_seed(42)
    K_PACKED = K // 2
    K_SCALE = K // 32

    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_packed = torch.randint(0, 16, (N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_scale = torch.randint(119, 123, (N, K_SCALE), device='cuda', dtype=torch.uint8)

    b_deq = dequant_ref(b_packed, b_scale, K)
    ref = a.float() @ b_deq.t()

    out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    triton_fused_fp4_matmul_nt(a, b_packed, b_scale, out, use_fp8=use_fp8)

    diff = (out.float() - ref).abs().max().item()
    rel = diff / (ref.abs().max().item() + 1e-12)
    status = "PASS ✅" if rel < 0.05 else "FAIL ❌"
    print(f"  {label} [{M}x{K}x{N}]: rel_err={rel:.4f} {status}")
    return rel < 0.05


def bench(M, K, N):
    torch.manual_seed(42)
    K_PACKED = K // 2
    K_SCALE = K // 32
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_packed = torch.randint(0, 16, (N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_scale = torch.randint(119, 123, (N, K_SCALE), device='cuda', dtype=torch.uint8)
    out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

    for mode, label in [(False, "float32"), (True, "dot_scaled")]:
        for _ in range(3):
            triton_fused_fp4_matmul_nt(a, b_packed, b_scale, out, use_fp8=mode)
        torch.cuda.synchronize()
        N_ITER = 20
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            triton_fused_fp4_matmul_nt(a, b_packed, b_scale, out, use_fp8=mode)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N_ITER * 1000
        print(f"  {label:12s} [{M}x{K}x{N}]: {ms:.2f} ms")


if __name__ == "__main__":
    print("=== Correctness ===")
    ok = True
    for M, K, N in [(16, 64, 16), (64, 256, 64), (384, 7168, 4096), (384, 2048, 7168)]:
        ok &= test(M, K, N, use_fp8=True, label="dot_scaled")
    for M, K, N in [(384, 7168, 4096)]:
        ok &= test(M, K, N, use_fp8=False, label="float32")

    print("\n=== Speed ===")
    for M, K, N in [(384, 7168, 4096), (384, 2048, 7168)]:
        bench(M, K, N)

    print(f"\n{'🎉 All passed!' if ok else '⚠️ Failures.'}")
