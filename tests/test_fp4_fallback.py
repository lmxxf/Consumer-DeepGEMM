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


if __name__ == "__main__":
    test_dequant_fp4_e2m1_with_e8m0_scale()
    test_fp8_fp4_gemm_nt_uses_fp4_dequant()
    test_m_grouped_fp8_fp4_gemm_nt_contiguous_uses_grouped_b()
