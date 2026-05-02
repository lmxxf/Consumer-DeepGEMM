import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from consumer_deep_gemm.gemm import _dequant_fp4_block, fp8_fp4_gemm_nt


def test_dequant_fp4_e2m1_with_e8m0_scale():
    packed = torch.tensor([[[0x21, 0x43]]], dtype=torch.uint8)
    scale = torch.tensor([[[127]]], dtype=torch.uint8)

    out = _dequant_fp4_block(packed, scale)

    expected = torch.tensor([[[0.5, 1.0, 1.5, 2.0]]], dtype=torch.bfloat16)
    assert torch.equal(out, expected)


def test_fp8_fp4_gemm_nt_uses_fp4_dequant():
    a = torch.ones((1, 4), dtype=torch.bfloat16)
    packed = torch.tensor([[0x21, 0x43]], dtype=torch.uint8)
    scale = torch.tensor([[127]], dtype=torch.uint8)
    d = torch.empty((1, 1), dtype=torch.float32)

    fp8_fp4_gemm_nt(a, (packed, scale), d)

    assert torch.equal(d, torch.tensor([[5.0]]))


if __name__ == "__main__":
    test_dequant_fp4_e2m1_with_e8m0_scale()
    test_fp8_fp4_gemm_nt_uses_fp4_dequant()
