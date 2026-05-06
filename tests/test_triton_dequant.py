"""Test Triton FP4 dequant against Python reference."""
import torch
import sys
sys.path.insert(0, '/work/Consumer-DeepGEMM')

from consumer_deep_gemm.gemm import _dequant_fp4_block, _dequant_fp8_block, _e8m0_to_float
from consumer_deep_gemm.triton_moe import triton_dequant_fp4, triton_dequant_fp8


def test_fp4_dequant():
    """Test triton FP4 dequant matches Python reference."""
    torch.manual_seed(42)
    N, K = 4096, 3584
    K_packed = K // 2

    b_packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device='cuda')
    b_scale = torch.randint(110, 140, (N, K // 32), dtype=torch.uint8, device='cuda')
    b_scale_f32 = _e8m0_to_float(b_scale)

    ref = _dequant_fp4_block(b_packed, b_scale_f32, block_k=32)
    out = triton_dequant_fp4(b_packed, b_scale_f32)

    diff = (ref.float() - out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"FP4 dequant [{N}, {K}]: max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}")
    assert max_diff < 0.01, f"FP4 dequant mismatch: max_diff={max_diff}"
    print("  PASS")


def test_fp4_dequant_shapes():
    """Test various shapes."""
    torch.manual_seed(42)
    for N, K in [(1, 64), (64, 128), (4096, 7168), (2048, 4096)]:
        K_packed = K // 2
        b_packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device='cuda')
        b_scale = torch.randint(110, 140, (N, K // 32), dtype=torch.uint8, device='cuda')
        b_scale_f32 = _e8m0_to_float(b_scale)

        ref = _dequant_fp4_block(b_packed, b_scale_f32, block_k=32)
        out = triton_dequant_fp4(b_packed, b_scale_f32)

        diff = (ref.float() - out.float()).abs()
        max_diff = diff.max().item()
        print(f"  [{N:5d}, {K:5d}]: max_diff={max_diff:.6f}", end="")
        assert max_diff < 0.01, f"FAIL: max_diff={max_diff}"
        print(" PASS")


def test_fp4_dequant_speed():
    """Benchmark triton vs python dequant."""
    import time
    torch.manual_seed(42)
    N, K = 4096, 3584
    K_packed = K // 2

    b_packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device='cuda')
    b_scale = torch.randint(110, 140, (N, K // 32), dtype=torch.uint8, device='cuda')
    b_scale_f32 = _e8m0_to_float(b_scale)

    # Warmup
    for _ in range(3):
        triton_dequant_fp4(b_packed, b_scale_f32)
        _dequant_fp4_block(b_packed, b_scale_f32, block_k=32)

    torch.cuda.synchronize()

    # Python reference
    t0 = time.perf_counter()
    for _ in range(20):
        _dequant_fp4_block(b_packed, b_scale_f32, block_k=32)
        torch.cuda.synchronize()
    t_py = (time.perf_counter() - t0) / 20

    # Triton
    t0 = time.perf_counter()
    for _ in range(20):
        triton_dequant_fp4(b_packed, b_scale_f32)
        torch.cuda.synchronize()
    t_tri = (time.perf_counter() - t0) / 20

    print(f"FP4 dequant [{N}, {K}]:")
    print(f"  Python:  {t_py*1000:.2f} ms")
    print(f"  Triton:  {t_tri*1000:.2f} ms")
    print(f"  Speedup: {t_py/t_tri:.1f}x")


if __name__ == '__main__':
    print("=== FP4 Dequant Correctness ===")
    test_fp4_dequant()

    print("\n=== FP4 Dequant Various Shapes ===")
    test_fp4_dequant_shapes()

    print("\n=== FP4 Dequant Speed ===")
    test_fp4_dequant_speed()

    print("\nAll tests passed!")
