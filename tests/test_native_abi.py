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


if __name__ == "__main__":
    test_native_grouped_fp8_fp4_abi_returns_none_then_fallback_runs()
    test_cutlass_mxfp8_mxfp4_grouped_can_implement_probe()
