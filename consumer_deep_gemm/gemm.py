"""GEMM operations matching DeepGEMM's Python API.

Phase 1: Pure PyTorch fallback (correct but slow)
Phase 2: CUTLASS SM120 kernels via C++ extension
"""

import torch
from typing import Optional

from . import native


_E2M1_TABLE = None


def _fp4_e2m1_table(device: torch.device) -> torch.Tensor:
    """OCP MXFP4 / NVFP4 E2M1 values used by DeepSeek V4 FP4 weights."""
    global _E2M1_TABLE
    if _E2M1_TABLE is None or _E2M1_TABLE.device != device:
        _E2M1_TABLE = torch.tensor(
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            ],
            dtype=torch.float32,
            device=device,
        )
    return _E2M1_TABLE


def _dequant_fp8_block(x: torch.Tensor, scale: torch.Tensor, block_k: int = 128) -> torch.Tensor:
    """Dequantize FP8 tensor with per-block scaling to BF16."""
    if x.dtype == torch.bfloat16 or x.dtype == torch.float32:
        return x
    orig_shape = x.shape
    M, K = orig_shape[0], orig_shape[1]
    n_blocks = (K + block_k - 1) // block_k
    pad = n_blocks * block_k - K

    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    x_f = x.reshape(M, n_blocks, block_k).to(torch.bfloat16)
    s = scale.to(torch.float32)
    if s.dim() == 2 and s.shape[1] == n_blocks:
        s = s.unsqueeze(2)
    x_f = x_f * s.to(torch.bfloat16)
    out = x_f.reshape(M, n_blocks * block_k)
    if pad:
        out = out[:, :K]
    return out.reshape(orig_shape).to(torch.bfloat16)


def _e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Convert uint8 E8M0 scales to float32 powers of two."""
    if scale.dtype == torch.uint8:
        return torch.pow(2.0, scale.to(torch.float32) - 127.0)
    return scale.to(torch.float32)


def _float_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Convert positive float scales to E8M0 bytes for CUTLASS MX kernels."""
    if scale.dtype == torch.uint8:
        return scale.contiguous()
    scale_f = scale.to(torch.float32)
    tiny = torch.finfo(torch.float32).tiny
    e8m0 = torch.ceil(torch.log2(torch.clamp(scale_f, min=tiny))) + 127.0
    return torch.clamp(e8m0, 0, 255).to(torch.uint8).contiguous()


def _expand_a_scale_for_cutlass_mx(scale: torch.Tensor) -> torch.Tensor:
    """Expand per-128-K activation scales to CUTLASS SM120 SFA layout.

    vLLM/DeepGEMM's MoE scatter produces activation scales as [M, K / 128].
    CUTLASS' SM120 block-scaled MX mainloop stores four scale atoms per
    logical activation scale in the SFA layout used by 72c/79c.
    """
    if scale.dim() == 2:
        return scale.repeat_interleave(4, dim=1).contiguous()
    return scale.contiguous()


def _native_mxfp8_mxfp4_args(a, b):
    if not (isinstance(a, tuple) and isinstance(b, tuple)):
        return a, b
    a_tensor, a_scale = a
    b_tensor, b_scale = b
    return (
        a_tensor,
        _expand_a_scale_for_cutlass_mx(_float_scale_to_e8m0(a_scale)),
    ), (
        b_tensor,
        _float_scale_to_e8m0(b_scale),
    )


def _dequant_fp4_block(x: torch.Tensor, scale: Optional[torch.Tensor], block_k: int = 32) -> torch.Tensor:
    """Dequantize packed E2M1 FP4 weights with per-32-value E8M0 scales.

    DeepSeek V4 stores two FP4 values per uint8. The scale tensor is normally
    one E8M0 byte per 32 values along K. This is a correctness fallback, not a
    performance path.
    """
    if x.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return x.to(torch.bfloat16)
    if x.dtype == torch.int8:
        x = x.view(torch.uint8)
    if x.dtype != torch.uint8:
        return x.to(torch.bfloat16)

    table = _fp4_e2m1_table(x.device)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    low_vals = table[low.to(torch.long)]
    high_vals = table[high.to(torch.long)]
    out = torch.stack((low_vals, high_vals), dim=-1).flatten(-2)

    if scale is None:
        return out.to(torch.bfloat16)

    k = out.shape[-1]
    n_blocks = (k + block_k - 1) // block_k
    pad = n_blocks * block_k - k
    if pad:
        out = torch.nn.functional.pad(out, (0, pad))

    scale_f = _e8m0_to_float(scale)
    while scale_f.dim() < out.dim():
        scale_f = scale_f.unsqueeze(-2)
    scale_f = scale_f.expand(*out.shape[:-1], n_blocks)

    out = out.reshape(*out.shape[:-1], n_blocks, block_k)
    out = out * scale_f.unsqueeze(-1)
    out = out.reshape(*out.shape[:-2], n_blocks * block_k)
    if pad:
        out = out[..., :k]
    return out.to(torch.bfloat16)


def _gemm_fallback(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = True) -> torch.Tensor:
    """Basic matmul fallback."""
    a_f = a.to(torch.bfloat16) if a.dtype not in (torch.bfloat16, torch.float32) else a
    b_f = b.to(torch.bfloat16) if b.dtype not in (torch.bfloat16, torch.float32) else b
    if transpose_b:
        return torch.mm(a_f, b_f.t()).to(torch.bfloat16)
    return torch.mm(a_f, b_f).to(torch.bfloat16)


def _normalize_fp8_gemm_args(a, sfa, b=None, sfb=None, d=None):
    """Accept both DeepGEMM and vLLM tuple-form FP8 GEMM arguments."""
    if isinstance(a, tuple) and isinstance(sfa, tuple):
        if b is None:
            raise TypeError("tuple-form fp8_gemm_nt requires output tensor")
        a_tensor, a_scale = a
        b_tensor, b_scale = sfa
        return a_tensor, a_scale, b_tensor, b_scale, b
    if b is None or sfb is None or d is None:
        raise TypeError("fp8_gemm_nt requires (a, sfa, b, sfb, d)")
    return a, sfa, b, sfb, d


def fp8_gemm_nt(a, sfa, b=None, sfb=None, d=None, c=None, **kwargs):
    """FP8 GEMM: D = A @ B^T, with per-block scale factors."""
    a, sfa, b, sfb, d = _normalize_fp8_gemm_args(a, sfa, b, sfb, d)
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
        b_deq = _dequant_fp4_block(b_tensor, b_scale)
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
        b_deq = _dequant_fp4_block(b_tensor, b_scale)
    else:
        b_deq = b.to(torch.bfloat16)

    result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32))
    if c is not None:
        result = result + c
    d.copy_(result)


def _dequant_fp8_arg(a) -> torch.Tensor:
    if isinstance(a, tuple):
        a_tensor, a_scale = a
        return _dequant_fp8_block(a_tensor, a_scale)
    return a.to(torch.bfloat16)


def _dequant_fp4_arg(b) -> torch.Tensor:
    if isinstance(b, tuple):
        b_tensor, b_scale = b
        return _dequant_fp4_block(b_tensor, b_scale)
    return b.to(torch.bfloat16)


def _grouped_segments_from_indices(m_indices: Optional[torch.Tensor], m: int, num_groups: int):
    """Return row indices per group for DeepGEMM contiguous grouped layout.

    DeepGEMM uses either a per-row group-id vector (normal contiguous layout)
    or a cumulative-end vector (psum layout). The per-row layout is the one
    vLLM uses for MoE routing; psum is kept as a correctness fallback.
    """
    if m_indices is None:
        if num_groups != 1:
            raise ValueError("m_indices is required when b has multiple groups")
        return [torch.arange(m)]

    if m_indices.numel() == m:
        return [(m_indices == group).nonzero(as_tuple=False).flatten() for group in range(num_groups)]

    if m_indices.numel() == num_groups:
        ends = m_indices.to("cpu", non_blocking=False).tolist()
        starts = [0] + [int(end) for end in ends[:-1]]
        return [
            torch.arange(start, int(end), device=m_indices.device)
            for start, end in zip(starts, ends)
        ]

    raise ValueError(
        f"m_indices must have length M ({m}) or num_groups ({num_groups}), "
        f"got {m_indices.numel()}"
    )


def _select_group_weight_nt(b_group: torch.Tensor, k: int) -> torch.Tensor:
    """Normalize a grouped B slice to [N, K] for NT matmul."""
    if b_group.dim() != 2:
        raise ValueError(f"grouped B slice must be 2D, got shape {tuple(b_group.shape)}")
    if b_group.shape[-1] == k:
        return b_group
    if b_group.shape[0] == k:
        return b_group.t().contiguous()
    raise ValueError(f"cannot infer B layout for shape {tuple(b_group.shape)} and K={k}")


def _m_grouped_fp8_fp4_fallback_nt(a, b, d, m_indices=None):
    a_deq = _dequant_fp8_arg(a)
    b_deq = _dequant_fp4_arg(b)

    if b_deq.dim() == 2:
        result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32).t())
        d.copy_(result)
        return

    if b_deq.dim() != 3:
        raise ValueError(f"grouped B must be 2D or 3D, got shape {tuple(b_deq.shape)}")

    m, k = a_deq.shape
    num_groups = b_deq.shape[0]
    d.zero_()
    for group, rows in enumerate(_grouped_segments_from_indices(m_indices, m, num_groups)):
        if rows.numel() == 0:
            continue
        rows = rows.to(device=a_deq.device)
        b_group = _select_group_weight_nt(b_deq[group], k)
        result = torch.mm(
            a_deq.index_select(0, rows).to(torch.float32),
            b_group.to(torch.float32).t(),
        )
        d.index_copy_(0, rows, result.to(d.dtype))


def m_grouped_fp8_gemm_nt_contiguous(a, sfa, b, sfb, d, m_indices=None, **kwargs):
    """M-grouped FP8 GEMM for MoE contiguous layout."""
    fp8_gemm_nt(a, sfa, b, sfb, d, **kwargs)


def m_grouped_fp8_gemm_nn_contiguous(a, sfa, b, sfb, d, m_indices=None, **kwargs):
    fp8_gemm_nn(a, sfa, b, sfb, d, **kwargs)


def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, m_indices=None, **kwargs):
    """M-grouped FP8×FP4 GEMM for MoE contiguous layout."""
    native_a, native_b = _native_mxfp8_mxfp4_args(a, b)
    native_result = native.m_grouped_fp8_fp4_gemm_nt_contiguous(
        native_a, native_b, d, m_indices, **kwargs
    )
    if native_result is not None:
        return
    _m_grouped_fp8_fp4_fallback_nt(a, b, d, m_indices)


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
