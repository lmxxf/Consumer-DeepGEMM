"""Einsum operations matching DeepGEMM's API."""

import torch
from typing import Optional


def einsum(expr: str, a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, **kwargs) -> None:
    """Generic einsum."""
    result = torch.einsum(expr, a.to(torch.float32), b.to(torch.float32))
    d.copy_(result)


def fp8_einsum(
    expr: str,
    a,
    b,
    d: torch.Tensor,
    **kwargs,
) -> None:
    """FP8 einsum with scale factors."""
    from .gemm import _dequant_fp8_block

    if isinstance(a, tuple):
        a_tensor, a_scale = a
        if a_tensor.dim() == 2:
            a_f = _dequant_fp8_block(a_tensor, a_scale).to(torch.float32)
        else:
            a_f = a_tensor.to(torch.float32) * a_scale.to(torch.float32)
    else:
        a_f = a.to(torch.float32)

    if isinstance(b, tuple):
        b_tensor, b_scale = b
        if b_tensor.dim() == 2:
            b_f = _dequant_fp8_block(b_tensor, b_scale).to(torch.float32)
        else:
            b_f = b_tensor.to(torch.float32) * b_scale.to(torch.float32)
    else:
        b_f = b.to(torch.float32)

    result = torch.einsum(expr, a_f, b_f)
    d.copy_(result)
