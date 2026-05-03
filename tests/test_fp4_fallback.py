import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from consumer_deep_gemm.gemm import (
    _dequant_fp4_block,
    fp8_fp4_gemm_nt,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
)


def test_dequant_fp4_e2m1_with_e8m0_scale():
    packed = torch.tensor([[[0x21, 0x43]]], dtype=torch.uint8)
    scale = torch.tensor([[[127]]], dtype=torch.uint8)

    out = _dequant_fp4_block(packed, scale)

    expected = torch.tensor([[[0.5, 1.0, 1.5, 2.0]]], dtype=torch.bfloat16)
    assert torch.equal(out, expected)


def test_fp8_fp4_gemm_nt_uses_fp4_dequant():
    a = torch.ones((1, 4), dtype=torch.bfloat16)
    packed = torch.tensor([[0x21, 0x43]], dtype=torch.uint8).view(torch.int8)
    scale = torch.tensor([[127]], dtype=torch.uint8)
    d = torch.empty((1, 1), dtype=torch.float32)

    fp8_fp4_gemm_nt(a, (packed, scale), d)

    assert torch.equal(d, torch.tensor([[5.0]]))


def test_m_grouped_fp8_fp4_gemm_nt_contiguous_uses_grouped_b():
    a = torch.ones((4, 4), dtype=torch.bfloat16)
    # group 0 dequantizes to [0.5, 1.0, 1.5, 2.0], group 1 to all 2.0.
    packed = torch.tensor(
        [
            [[0x21, 0x43]],
            [[0x44, 0x44]],
        ],
        dtype=torch.uint8,
    )
    scale = torch.full((2, 1, 1), 127, dtype=torch.uint8)
    groups = torch.tensor([0, 1, -1, 0], dtype=torch.int32)
    d = torch.empty((4, 1), dtype=torch.bfloat16)

    m_grouped_fp8_fp4_gemm_nt_contiguous(a, (packed, scale), d, groups)

    expected = torch.tensor([[5.0], [8.0], [0.0], [5.0]], dtype=torch.bfloat16)
    assert torch.equal(d, expected)


def test_m_grouped_fp8_fp4_gemm_nt_contiguous_interleaved_experts():
    """Expert IDs [1, 0, 1, 0] — non-contiguous, must still give correct results."""
    a = torch.ones((4, 4), dtype=torch.bfloat16)
    packed = torch.tensor(
        [
            [[0x21, 0x43]],  # group 0: [0.5, 1.0, 1.5, 2.0] -> sum=5
            [[0x44, 0x44]],  # group 1: [2.0, 2.0, 2.0, 2.0] -> sum=8
        ],
        dtype=torch.uint8,
    )
    scale = torch.full((2, 1, 1), 127, dtype=torch.uint8)
    groups = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    d = torch.empty((4, 1), dtype=torch.bfloat16)

    m_grouped_fp8_fp4_gemm_nt_contiguous(a, (packed, scale), d, groups)

    expected = torch.tensor([[8.0], [5.0], [8.0], [5.0]], dtype=torch.bfloat16)
    assert torch.equal(d, expected), f"got {d} expected {expected}"


def test_m_grouped_fp8_gemm_nt_contiguous_uses_grouped_b():
    from consumer_deep_gemm.gemm import m_grouped_fp8_gemm_nt_contiguous, _dequant_fp8_block

    a = torch.ones((4, 4), dtype=torch.bfloat16)
    sfa = torch.ones((4, 1), dtype=torch.float32)
    b = torch.stack([
        torch.full((1, 4), 1.0, dtype=torch.bfloat16),
        torch.full((1, 4), 2.0, dtype=torch.bfloat16),
    ])
    sfb = torch.ones((2, 1, 1), dtype=torch.float32)
    groups = torch.tensor([0, 1, -1, 0], dtype=torch.int32)
    d = torch.empty((4, 1), dtype=torch.bfloat16)

    m_grouped_fp8_gemm_nt_contiguous(a, sfa, b, sfb, d, groups)

    expected = torch.tensor([[4.0], [8.0], [0.0], [4.0]], dtype=torch.bfloat16)
    assert torch.equal(d, expected), f"got {d} expected {expected}"


def test_m_grouped_fp8_fp4_gemm_nt_masked():
    from consumer_deep_gemm.gemm import m_grouped_fp8_fp4_gemm_nt_masked

    a = torch.ones((4, 4), dtype=torch.bfloat16)
    packed = torch.tensor(
        [
            [[0x21, 0x43]],  # group 0: sum=5
            [[0x44, 0x44]],  # group 1: sum=8
        ],
        dtype=torch.uint8,
    )
    scale = torch.full((2, 1, 1), 127, dtype=torch.uint8)
    masked_m = torch.tensor([1, 2], dtype=torch.int32)
    d = torch.empty((4, 1), dtype=torch.bfloat16)

    m_grouped_fp8_fp4_gemm_nt_masked(a, (packed, scale), d, masked_m)

    assert d[0].item() == 5.0, f"row 0 = {d[0].item()}"
    assert d[2].item() == 8.0, f"row 2 = {d[2].item()}"
    assert d[3].item() == 8.0, f"row 3 = {d[3].item()}"
    assert d[1].item() == 0.0, f"row 1 should be zero, got {d[1].item()}"


if __name__ == "__main__":
    test_dequant_fp4_e2m1_with_e8m0_scale()
    test_fp8_fp4_gemm_nt_uses_fp4_dequant()
    test_m_grouped_fp8_fp4_gemm_nt_contiguous_uses_grouped_b()
    test_m_grouped_fp8_fp4_gemm_nt_contiguous_interleaved_experts()
    test_m_grouped_fp8_gemm_nt_contiguous_uses_grouped_b()
    test_m_grouped_fp8_fp4_gemm_nt_masked()
    print("all tests passed")
