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
    if scale.dim() == 2 and scale.shape[1] > 1:
        block_k = K // scale.shape[1]
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
    """Convert positive float scales to E8M0 bytes for CUTLASS MX kernels.

    E8M0 encodes 2^(e-127). For exact powers of two this is lossless.
    For non-power-of-two scales we round to nearest exponent.
    """
    if scale.dtype == torch.uint8:
        return scale.contiguous()
    scale_f = scale.to(torch.float32)
    tiny = torch.finfo(torch.float32).tiny
    e8m0 = torch.round(torch.log2(torch.clamp(scale_f, min=tiny))) + 127.0
    return torch.clamp(e8m0, 0, 255).to(torch.uint8).contiguous()


def _reorder_scale_to_cutlass_sf_atom(scale: torch.Tensor, m: int, k_blocks: int) -> torch.Tensor:
    """Reorder row-major [M, K/SFVecSize] scale to CUTLASS SM120 SfAtom layout.

    The K-major SfAtom for SM120 block-scaled MX is:
        Shape:  ((32, 4), (SFVecSize, 4))
        Stride: ((16, 4), (0, 1))

    This means within each 128-row × 4-K_block tile, the data is stored as:
        for each of 4 k_in_4 values:
            for each of 4 m_32 blocks (of 32 rows each):
                scale[m_32, k_in_4] (broadcast across 32 rows within block)

    Since stride-0 dimensions are broadcast, the actual unique data per
    128×4 tile is 4×4=16 values, stored in the order:
        [k0_m0, k0_m1, k0_m2, k0_m3, k1_m0, k1_m1, k1_m2, k1_m3, ...]

    But the full tile has 128×4=512 "slots" with broadcast. The actual stored
    data (filter_zeros) has size = ceil(M/32) * ceil(K_blocks/4) * 4.
    We just need to interleave the scales in the right order for the 16-element
    tile atom: groups of 4 m-chunks interleaved with k-blocks-of-4.
    """
    if scale.dim() != 2:
        return scale.contiguous()

    M, Kb = scale.shape
    m_tiles = (M + 127) // 128
    k_tiles = (Kb + 3) // 4

    pad_m = m_tiles * 128 - M
    pad_k = k_tiles * 4 - Kb
    if pad_m > 0 or pad_k > 0:
        scale = torch.nn.functional.pad(scale, (0, pad_k, 0, pad_m), value=127)

    # [m_tiles, 128, k_tiles, 4] -> split 128 into (4 groups of 32)
    scale = scale.reshape(m_tiles, 4, 32, k_tiles, 4)
    # Atom offset = m_in_32 * 16 + m_32 * 4 + k_in_4
    # Target contiguous dim order: [m_tiles, k_tiles, m_in_32, m_32, k_in_4]
    # strides: [512*k_tiles, 512, 16, 4, 1] — matches SfAtom
    scale = scale.permute(0, 3, 2, 1, 4).contiguous()

    return scale.reshape(-1).contiguous()


def _prepare_sfa_for_native(scale_e8m0: torch.Tensor, sfvec_ratio: int = 4) -> torch.Tensor:
    """Expand per-128-K E8M0 activation scale to per-32-K and keep as 2D [M, K/32].

    Scale layout reordering is done per-group in C++ launch to handle grouped
    pointer offsets correctly.
    """
    if scale_e8m0.dim() == 2:
        return scale_e8m0.repeat_interleave(sfvec_ratio, dim=1).contiguous()
    return scale_e8m0.contiguous()


def _native_mxfp8_mxfp4_args(a, b):
    if not (isinstance(a, tuple) and isinstance(b, tuple)):
        return a, b
    a_tensor, a_scale = a
    b_tensor, b_scale = b
    a_scale_e8m0 = _float_scale_to_e8m0(a_scale)
    a_scale_e8m0 = _prepare_sfa_for_native(a_scale_e8m0)
    b_scale_e8m0 = _float_scale_to_e8m0(b_scale)
    return (
        a_tensor,
        a_scale_e8m0,
    ), (
        b_tensor,
        b_scale_e8m0,
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


def _m_grouped_fp8_fallback(a, sfa, b, sfb, d, m_indices, transpose_b=True):
    a_deq = _dequant_fp8_block(a, sfa)
    b_deq = _dequant_fp8_block(b, sfb)

    if b_deq.dim() == 2:
        if transpose_b:
            result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32).t())
        else:
            result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32))
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
        a_group = a_deq.index_select(0, rows).to(torch.float32)
        b_group = b_deq[group].to(torch.float32)
        if transpose_b:
            b_group = _select_group_weight_nt(b_group, k)
            result = torch.mm(a_group, b_group.t())
        else:
            result = torch.mm(a_group, b_group)
        d.index_copy_(0, rows, result.to(d.dtype))


def m_grouped_fp8_gemm_nt_contiguous(a, sfa, b, sfb, d, m_indices=None, **kwargs):
    """M-grouped FP8 GEMM for MoE contiguous layout."""
    _m_grouped_fp8_fallback(a, sfa, b, sfb, d, m_indices, transpose_b=True)


def m_grouped_fp8_gemm_nn_contiguous(a, sfa, b, sfb, d, m_indices=None, **kwargs):
    _m_grouped_fp8_fallback(a, sfa, b, sfb, d, m_indices, transpose_b=False)


def _m_grouped_fp8_fp4_dequant_mm_nt(a, b, d, m_indices):
    """FP4 dequant + torch.mm path — bypasses CUTLASS launch overhead.

    Only dequantizes the active experts (typically 6 out of 256),
    then uses cuBLAS BF16 matmul which handles small M efficiently.
    """
    a_deq = _dequant_fp8_arg(a)

    if isinstance(b, tuple):
        b_tensor, b_scale = b
    else:
        b_tensor = b
        b_scale = None
    if b_tensor.dtype == torch.int8:
        b_tensor = b_tensor.view(torch.uint8)

    if b_tensor.dim() != 3:
        b_deq = _dequant_fp4_block(b_tensor, b_scale)
        result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32).t())
        d.copy_(result.to(d.dtype))
        return

    m, k = a_deq.shape
    num_groups = b_tensor.size(0)

    if m_indices is None or m_indices.numel() != m:
        _m_grouped_fp8_fp4_fallback_nt(a, b, d, m_indices)
        return

    active_groups = m_indices[m_indices >= 0].unique()
    if active_groups.numel() == 0:
        d.zero_()
        return

    d.zero_()
    for gid_t in active_groups:
        gid = gid_t.item()
        mask = (m_indices == gid)
        rows = mask.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue

        b_group_scale = None
        if b_scale is not None:
            if b_scale.dim() == 3:
                b_group_scale = b_scale[gid]
            else:
                b_group_scale = b_scale
        b_group_deq = _dequant_fp4_block(b_tensor[gid], b_group_scale)
        b_group_deq = _select_group_weight_nt(b_group_deq, k)

        a_group = a_deq.index_select(0, rows).to(torch.float32)
        result = torch.mm(a_group, b_group_deq.to(torch.float32).t())
        d.index_copy_(0, rows, result.to(d.dtype))


def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, m_indices=None, **kwargs):
    """M-grouped FP8×FP4 GEMM for MoE contiguous layout."""
    _m_grouped_fp8_fp4_dequant_mm_nt(a, b, d, m_indices)


def m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, m_indices=None, **kwargs):
    a_deq = _dequant_fp8_arg(a)
    b_deq = _dequant_fp4_arg(b)

    if b_deq.dim() == 2:
        result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32))
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
        result = torch.mm(
            a_deq.index_select(0, rows).to(torch.float32),
            b_deq[group].to(torch.float32),
        )
        d.index_copy_(0, rows, result.to(d.dtype))


def m_grouped_fp8_gemm_nt_masked(a, sfa, b, sfb, d, masked_m=None, **kwargs):
    """M-grouped FP8 GEMM with masking for MoE.

    masked_m is a 1D tensor of per-group row counts. Each group g uses
    rows [g*block : g*block + masked_m[g]] from A and writes to the
    corresponding rows of D.
    """
    a_deq = _dequant_fp8_block(a, sfa)
    b_deq = _dequant_fp8_block(b, sfb)

    if b_deq.dim() != 3 or masked_m is None:
        result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32).t())
        d.copy_(result)
        return

    num_groups = b_deq.shape[0]
    block = a_deq.shape[0] // num_groups
    masked_m_cpu = masked_m.to("cpu", non_blocking=False).tolist()
    d.zero_()
    for g in range(num_groups):
        count = int(masked_m_cpu[g])
        if count <= 0:
            continue
        row_start = g * block
        a_g = a_deq[row_start:row_start + count].to(torch.float32)
        b_g = b_deq[g].to(torch.float32)
        result = torch.mm(a_g, b_g.t())
        d[row_start:row_start + count].copy_(result.to(d.dtype))


def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m=None, **kwargs):
    """M-grouped FP8×FP4 GEMM with masking for MoE."""
    a_deq = _dequant_fp8_arg(a)
    b_deq = _dequant_fp4_arg(b)

    if b_deq.dim() != 3 or masked_m is None:
        result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32).t())
        d.copy_(result)
        return

    num_groups = b_deq.shape[0]
    block = a_deq.shape[0] // num_groups
    masked_m_cpu = masked_m.to("cpu", non_blocking=False).tolist()
    d.zero_()
    for g in range(num_groups):
        count = int(masked_m_cpu[g])
        if count <= 0:
            continue
        row_start = g * block
        a_g = a_deq[row_start:row_start + count].to(torch.float32)
        b_g = b_deq[g].to(torch.float32)
        result = torch.mm(a_g, b_g.t())
        d[row_start:row_start + count].copy_(result.to(d.dtype))


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


def _m_grouped_bf16_fallback(a, b, d, m_indices, transpose_b=True):
    if b.dim() == 2:
        if transpose_b:
            result = torch.mm(a.to(torch.float32), b.to(torch.float32).t())
        else:
            result = torch.mm(a.to(torch.float32), b.to(torch.float32))
        d.copy_(result)
        return

    if b.dim() != 3:
        raise ValueError(f"grouped B must be 2D or 3D, got shape {tuple(b.shape)}")

    m, k = a.shape
    num_groups = b.shape[0]
    d.zero_()
    for group, rows in enumerate(_grouped_segments_from_indices(m_indices, m, num_groups)):
        if rows.numel() == 0:
            continue
        rows = rows.to(device=a.device)
        a_group = a.index_select(0, rows).to(torch.float32)
        b_group = b[group].to(torch.float32)
        if transpose_b:
            result = torch.mm(a_group, b_group.t())
        else:
            result = torch.mm(a_group, b_group)
        d.index_copy_(0, rows, result.to(d.dtype))


def m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices=None, **kwargs):
    _m_grouped_bf16_fallback(a, b, d, m_indices, transpose_b=True)


def m_grouped_bf16_gemm_nn_contiguous(a, b, d, m_indices=None, **kwargs):
    _m_grouped_bf16_fallback(a, b, d, m_indices, transpose_b=False)


def m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m=None, **kwargs):
    if b.dim() != 3 or masked_m is None:
        bf16_gemm_nt(a, b, d, **kwargs)
        return

    num_groups = b.shape[0]
    block = a.shape[0] // num_groups
    masked_m_cpu = masked_m.to("cpu", non_blocking=False).tolist()
    d.zero_()
    for g in range(num_groups):
        count = int(masked_m_cpu[g])
        if count <= 0:
            continue
        row_start = g * block
        a_g = a[row_start:row_start + count].to(torch.float32)
        b_g = b[g].to(torch.float32)
        result = torch.mm(a_g, b_g.t())
        d[row_start:row_start + count].copy_(result.to(d.dtype))


def cublaslt_gemm_nt(a, b, d, **kwargs):
    """cuBLASLt GEMM fallback."""
    result = torch.mm(a.to(torch.float32), b.to(torch.float32).t())
    d.copy_(result)
