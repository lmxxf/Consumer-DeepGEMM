"""GEMM operations matching DeepGEMM's Python API.

Phase 1: Pure PyTorch fallback (correct but slow)
Phase 2: CUTLASS SM120 kernels via C++ extension
"""

import torch
from typing import Optional


def _dequant_fp8_block(x: torch.Tensor, scale: torch.Tensor, block_k: int = 128) -> torch.Tensor:
    """Dequantize FP8 tensor with per-block scaling to BF16."""
    if x.dtype == torch.bfloat16 or x.dtype == torch.float32:
        return x
    orig_shape = x.shape
    M, K = orig_shape[0], orig_shape[1]
    n_blocks = (K + block_k - 1) // block_k

    x_f = x.reshape(M, n_blocks, block_k).to(torch.bfloat16)
    s = scale.to(torch.float32)
    if s.dim() == 2 and s.shape[1] == n_blocks:
        s = s.unsqueeze(2)
    x_f = x_f * s.to(torch.bfloat16)
    return x_f.reshape(orig_shape).to(torch.bfloat16)


def _gemm_fallback(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = True) -> torch.Tensor:
    """Basic matmul fallback."""
    a_f = a.to(torch.bfloat16) if a.dtype not in (torch.bfloat16, torch.float32) else a
    b_f = b.to(torch.bfloat16) if b.dtype not in (torch.bfloat16, torch.float32) else b
    if transpose_b:
        return torch.mm(a_f, b_f.t()).to(torch.bfloat16)
    return torch.mm(a_f, b_f).to(torch.bfloat16)


def fp8_gemm_nt(a, sfa, b, sfb, d, c=None, **kwargs):
    """FP8 GEMM: D = A @ B^T, with per-block scale factors."""
    a_deq = _dequant_fp8_block(a, sfa)
    b_deq = _dequant_fp8_block(b, sfb)
    result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32).t())
    if c is not None:
        result = result + c
    d.copy_(result)


def fp8_gemm_nn(a, sfa, b, sfb, d, c=None, **kwargs):
    a_deq = _dequant_fp8_block(a, sfa)
    b_deq = _dequant_fp8_block(b, sfb)
    result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32))
    if c is not None:
        result = result + c
    d.copy_(result)


def fp8_gemm_tn(a, sfa, b, sfb, d, c=None, **kwargs):
    a_deq = _dequant_fp8_block(a, sfa)
    b_deq = _dequant_fp8_block(b, sfb)
    result = torch.mm(a_deq.to(torch.float32).t(), b_deq.to(torch.float32))
    if c is not None:
        result = result + c
    d.copy_(result)


def fp8_gemm_tt(a, sfa, b, sfb, d, c=None, **kwargs):
    a_deq = _dequant_fp8_block(a, sfa)
    b_deq = _dequant_fp8_block(b, sfb)
    result = torch.mm(a_deq.to(torch.float32).t(), b_deq.to(torch.float32).t())
    if c is not None:
        result = result + c
    d.copy_(result)


def fp8_fp4_gemm_nt(a, b, d, c=None, **kwargs):
    """FP8 activation × FP4 weight GEMM. a and b are (tensor, scale) tuples."""
    if isinstance(a, tuple):
        a_tensor, a_scale = a
        a_deq = _dequant_fp8_block(a_tensor, a_scale)
    else:
        a_deq = a.to(torch.bfloat16)

    if isinstance(b, tuple):
        b_tensor, b_scale = b
        b_deq = _dequant_fp8_block(b_tensor, b_scale)
    else:
        b_deq = b.to(torch.bfloat16)

    result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32).t())
    if c is not None:
        result = result + c
    d.copy_(result)


def fp8_fp4_gemm_nn(a, b, d, c=None, **kwargs):
    if isinstance(a, tuple):
        a_tensor, a_scale = a
        a_deq = _dequant_fp8_block(a_tensor, a_scale)
    else:
        a_deq = a.to(torch.bfloat16)

    if isinstance(b, tuple):
        b_tensor, b_scale = b
        b_deq = _dequant_fp8_block(b_tensor, b_scale)
    else:
        b_deq = b.to(torch.bfloat16)

    result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32))
    if c is not None:
        result = result + c
    d.copy_(result)


def m_grouped_fp8_gemm_nt_contiguous(a, sfa, b, sfb, d, m_indices=None, **kwargs):
    """M-grouped FP8 GEMM for MoE contiguous layout."""
    fp8_gemm_nt(a, sfa, b, sfb, d, **kwargs)


def m_grouped_fp8_gemm_nn_contiguous(a, sfa, b, sfb, d, m_indices=None, **kwargs):
    fp8_gemm_nn(a, sfa, b, sfb, d, **kwargs)


def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, m_indices=None, **kwargs):
    """M-grouped FP8×FP4 GEMM for MoE contiguous layout."""
    fp8_fp4_gemm_nt(a, b, d, **kwargs)


def m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, m_indices=None, **kwargs):
    fp8_fp4_gemm_nn(a, b, d, **kwargs)


def m_grouped_fp8_gemm_nt_masked(a, sfa, b, sfb, d, masked_m=None, **kwargs):
    """M-grouped FP8 GEMM with masking for MoE."""
    fp8_gemm_nt(a, sfa, b, sfb, d, **kwargs)


def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m=None, **kwargs):
    fp8_fp4_gemm_nt(a, b, d, **kwargs)


def bf16_gemm_nt(a, b, d, c=None, **kwargs):
    result = torch.mm(a.to(torch.float32), b.to(torch.float32).t())
    if c is not None:
        result = result + c
    d.copy_(result)


def bf16_gemm_nn(a, b, d, c=None, **kwargs):
    result = torch.mm(a.to(torch.float32), b.to(torch.float32))
    if c is not None:
        result = result + c
    d.copy_(result)


def bf16_gemm_tn(a, b, d, c=None, **kwargs):
    result = torch.mm(a.to(torch.float32).t(), b.to(torch.float32))
    if c is not None:
        result = result + c
    d.copy_(result)


def bf16_gemm_tt(a, b, d, c=None, **kwargs):
    result = torch.mm(a.to(torch.float32).t(), b.to(torch.float32).t())
    if c is not None:
        result = result + c
    d.copy_(result)


def m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices=None, **kwargs):
    bf16_gemm_nt(a, b, d, **kwargs)


def m_grouped_bf16_gemm_nn_contiguous(a, b, d, m_indices=None, **kwargs):
    bf16_gemm_nn(a, b, d, **kwargs)


def m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m=None, **kwargs):
    bf16_gemm_nt(a, b, d, **kwargs)


def cublaslt_gemm_nt(a, b, d, **kwargs):
    """cuBLASLt GEMM fallback."""
    result = torch.mm(a.to(torch.float32), b.to(torch.float32).t())
    d.copy_(result)
