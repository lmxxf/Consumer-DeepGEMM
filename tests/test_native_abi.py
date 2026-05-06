import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import consumer_deep_gemm as dg
from consumer_deep_gemm import native


def test_native_grouped_fp8_fp4_abi_returns_none_then_fallback_runs():
    if not torch.cuda.is_available() or not native.is_available():
        return

    a = torch.ones((4, 4), device="cuda", dtype=torch.bfloat16)
    a_scale = torch.ones((4, 1), device="cuda")
    packed = torch.tensor(
        [
            [[0x21, 0x43]],
            [[0x44, 0x44]],
        ],
        device="cuda",
        dtype=torch.uint8,
    ).view(torch.int8)
    scale = torch.full((2, 1, 1), 127, device="cuda", dtype=torch.uint8)
    groups = torch.tensor([0, 1, -1, 0], device="cuda", dtype=torch.int32)
    d = torch.empty((4, 1), device="cuda", dtype=torch.bfloat16)

    assert native.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a, a_scale),
        (packed, scale),
        d,
        groups,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    ) is None

    dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a, a_scale),
        (packed, scale),
        d,
        groups,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )
    torch.cuda.synchronize()

    expected = torch.tensor([[5.0], [8.0], [0.0], [5.0]], dtype=torch.bfloat16)
    assert torch.equal(d.cpu(), expected)


def test_cutlass_mxfp8_mxfp4_grouped_can_implement_probe():
    if not torch.cuda.is_available() or not native.is_available():
        return

    a = torch.empty((128, 128), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty((2, 128, 64), device="cuda", dtype=torch.int8)
    d = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)

    assert native.cutlass_mxfp8_mxfp4_can_implement_probe(a, b, d) is True


def test_native_grouped_fp8_fp4_launch_smoke():
    if not torch.cuda.is_available() or not native.is_available():
        return

    a = torch.zeros((128, 128), device="cuda", dtype=torch.float8_e4m3fn)
    a_scale = torch.full((128, 4), 127, device="cuda", dtype=torch.uint8)
    b = torch.zeros((2, 128, 64), device="cuda", dtype=torch.int8)
    b_scale = torch.full((2, 128, 4), 127, device="cuda", dtype=torch.uint8)
    d = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    # Per-row expert_ids format (vLLM contiguous layout)
    m_indices = torch.full((128,), -1, device="cuda", dtype=torch.int32)
    m_indices[:64] = 0
    m_indices[64:128] = 1

    launched = native.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a, a_scale),
        (b, b_scale),
        d,
        m_indices,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )
    torch.cuda.synchronize()

    assert launched is True
    assert torch.equal(d.cpu(), torch.zeros_like(d).cpu())


def test_public_grouped_fp8_fp4_converts_float_scales_for_native():
    if not torch.cuda.is_available() or not native.is_available():
        return

    a = torch.zeros((128, 128), device="cuda", dtype=torch.float8_e4m3fn)
    a_scale = torch.ones((128, 4), device="cuda", dtype=torch.float32)
    b = torch.zeros((2, 128, 64), device="cuda", dtype=torch.int8)
    b_scale = torch.ones((2, 128, 4), device="cuda", dtype=torch.float32)
    d = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    m_indices = torch.full((128,), -1, device="cuda", dtype=torch.int32)
    m_indices[:64] = 0
    m_indices[64:128] = 1

    assert dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a, a_scale),
        (b, b_scale),
        d,
        m_indices,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    ) is None
    torch.cuda.synchronize()

    assert torch.equal(d.cpu(), torch.zeros_like(d).cpu())


def test_public_grouped_fp8_fp4_accepts_vllm_expert_ids_with_padding():
    if not torch.cuda.is_available() or not native.is_available():
        return

    a = torch.zeros((256, 128), device="cuda", dtype=torch.float8_e4m3fn)
    a_scale = torch.ones((256, 1), device="cuda", dtype=torch.float32)
    b = torch.zeros((2, 128, 64), device="cuda", dtype=torch.int8)
    b_scale = torch.ones((2, 128, 4), device="cuda", dtype=torch.float32)
    d = torch.empty((256, 128), device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.full((256,), -1, device="cuda", dtype=torch.int32)
    expert_ids[:17] = 0
    expert_ids[128:151] = 1

    assert dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a, a_scale),
        (b, b_scale),
        d,
        expert_ids,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    ) is None
    torch.cuda.synchronize()

    assert torch.equal(d.cpu(), torch.zeros_like(d).cpu())


def test_public_grouped_fp8_fp4_nonzero_reference_case():
    if not torch.cuda.is_available() or not native.is_available():
        return

    a = torch.ones((128, 128), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale = torch.ones((128, 1), device="cuda", dtype=torch.float32)
    b = torch.full((1, 128, 64), 0x22, device="cuda", dtype=torch.uint8).view(torch.int8)
    b_scale = torch.ones((1, 128, 4), device="cuda", dtype=torch.float32)
    d = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.zeros((128,), device="cuda", dtype=torch.int32)

    dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a, a_scale),
        (b, b_scale),
        d,
        expert_ids,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )
    torch.cuda.synchronize()

    expected = torch.full_like(d, 128)
    assert torch.equal(d.cpu(), expected.cpu())


def test_public_grouped_fp8_fp4_interleaved_expert_ids():
    """Non-contiguous expert_ids [1,0,1,0] — must still produce correct results via scatter/gather."""
    if not torch.cuda.is_available() or not native.is_available():
        return

    a = torch.ones((4, 4), device="cuda", dtype=torch.bfloat16)
    packed = torch.tensor(
        [
            [[0x21, 0x43]],  # group 0: dequant to [0.5, 1.0, 1.5, 2.0] -> sum=5
            [[0x44, 0x44]],  # group 1: dequant to [2.0, 2.0, 2.0, 2.0] -> sum=8
        ],
        device="cuda",
        dtype=torch.uint8,
    ).view(torch.int8)
    scale = torch.full((2, 1, 1), 127, device="cuda", dtype=torch.uint8)
    groups = torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32)
    d = torch.empty((4, 1), device="cuda", dtype=torch.bfloat16)

    dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a, torch.ones((4, 1), device="cuda")),
        (packed, scale),
        d,
        groups,
    )
    torch.cuda.synchronize()

    expected = torch.tensor([[8.0], [5.0], [8.0], [5.0]], dtype=torch.bfloat16)
    assert torch.equal(d.cpu(), expected), f"got {d.cpu()} expected {expected}"


if __name__ == "__main__":
    test_native_grouped_fp8_fp4_abi_returns_none_then_fallback_runs()
    test_cutlass_mxfp8_mxfp4_grouped_can_implement_probe()
    test_native_grouped_fp8_fp4_launch_smoke()
    test_public_grouped_fp8_fp4_converts_float_scales_for_native()
    test_public_grouped_fp8_fp4_accepts_vllm_expert_ids_with_padding()
    test_public_grouped_fp8_fp4_nonzero_reference_case()
    test_public_grouped_fp8_fp4_interleaved_expert_ids()
    print("all tests passed")
