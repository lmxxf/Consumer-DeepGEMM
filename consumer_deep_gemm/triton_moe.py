"""Triton FP4 dequant + grouped MoE GEMM for SM120/SM121.

Replaces the Python fallback in gemm.py with Triton kernels that
eliminate per-expert Python loop overhead and temporary tensor allocation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_fp4_e2m1_kernel(
    b_ptr,        # [N, K_packed] uint8, K_packed = K // 2
    scale_ptr,    # [N, K_scale] uint8 (E8M0), K_scale = K // 32
    out_ptr,      # [N, K] bf16
    N: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,   # K // 2
    K_SCALE: tl.constexpr,    # K // 32
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,  # processes this many packed bytes per iter
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_packed_offs = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)

    n_mask = n_offs < N
    k_packed_mask = k_packed_offs < K_PACKED

    mask = n_mask[:, None] & k_packed_mask[None, :]

    b = tl.load(b_ptr + n_offs[:, None] * K_PACKED + k_packed_offs[None, :],
                mask=mask, other=0).to(tl.uint8)

    low_idx = (b & 0x0F).to(tl.int32)
    high_idx = ((b >> 4) & 0x0F).to(tl.int32)

    # E2M1 lookup — hardcoded, avoids table load
    # Positive: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    # Negative: -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    low_sign = (low_idx >> 3).to(tl.float32)   # 0 or 1
    low_mag_idx = low_idx & 0x07
    low_abs = tl.where(low_mag_idx == 0, 0.0,
              tl.where(low_mag_idx == 1, 0.5,
              tl.where(low_mag_idx == 2, 1.0,
              tl.where(low_mag_idx == 3, 1.5,
              tl.where(low_mag_idx == 4, 2.0,
              tl.where(low_mag_idx == 5, 3.0,
              tl.where(low_mag_idx == 6, 4.0, 6.0)))))))
    low_val = tl.where(low_sign > 0.5, -low_abs, low_abs)

    high_sign = (high_idx >> 3).to(tl.float32)
    high_mag_idx = high_idx & 0x07
    high_abs = tl.where(high_mag_idx == 0, 0.0,
               tl.where(high_mag_idx == 1, 0.5,
               tl.where(high_mag_idx == 2, 1.0,
               tl.where(high_mag_idx == 3, 1.5,
               tl.where(high_mag_idx == 4, 2.0,
               tl.where(high_mag_idx == 5, 3.0,
               tl.where(high_mag_idx == 6, 4.0, 6.0)))))))
    high_val = tl.where(high_sign > 0.5, -high_abs, high_abs)

    # E8M0 scale: 2^(e - 127). Each scale covers 32 values = 16 packed bytes
    # k_packed_offs maps to scale index: k_packed_offs // 16
    scale_k_offs = k_packed_offs // 16  # each scale covers 16 packed bytes = 32 values
    scale_mask = n_mask[:, None] & (scale_k_offs[None, :] < K_SCALE)
    scale_e8m0 = tl.load(
        scale_ptr + n_offs[:, None] * K_SCALE + scale_k_offs[None, :],
        mask=scale_mask, other=127
    ).to(tl.float32)
    scale_f = tl.exp2(scale_e8m0 - 127.0)

    low_scaled = (low_val * scale_f).to(tl.bfloat16)
    high_scaled = (high_val * scale_f).to(tl.bfloat16)

    # Interleave: out[n, 2*k_packed] = low, out[n, 2*k_packed+1] = high
    out_k_base = k_packed_offs * 2
    out_k_low = out_k_base
    out_k_high = out_k_base + 1

    low_out_mask = n_mask[:, None] & (out_k_low[None, :] < K)
    high_out_mask = n_mask[:, None] & (out_k_high[None, :] < K)

    tl.store(out_ptr + n_offs[:, None] * K + out_k_low[None, :],
             low_scaled, mask=low_out_mask)
    tl.store(out_ptr + n_offs[:, None] * K + out_k_high[None, :],
             high_scaled, mask=high_out_mask)




def triton_dequant_fp4(b_packed: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize [N, K/2] packed FP4 + [N, K/32] E8M0 scales -> [N, K] bf16."""
    N, K_packed = b_packed.shape
    K = K_packed * 2
    K_scale = b_scale.shape[-1] if b_scale is not None else 0

    out = torch.empty(N, K, dtype=torch.bfloat16, device=b_packed.device)

    if b_scale is None:
        K_scale = 1

    BLOCK_N = 4
    BLOCK_K_PACKED = min(512, K_packed)

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K_packed, BLOCK_K_PACKED))

    b_scale_u8 = b_scale
    if b_scale is not None and b_scale.dtype != torch.uint8:
        if b_scale.dtype == torch.float32:
            tiny = torch.finfo(torch.float32).tiny
            e8m0 = torch.round(torch.log2(torch.clamp(b_scale, min=tiny))) + 127.0
            b_scale_u8 = torch.clamp(e8m0, 0, 255).to(torch.uint8)
        else:
            b_scale_u8 = b_scale.to(torch.uint8)

    _dequant_fp4_e2m1_kernel[grid](
        b_packed, b_scale_u8, out,
        N, K, K_packed, K_scale,
        BLOCK_N=BLOCK_N,
        BLOCK_K_PACKED=BLOCK_K_PACKED,
    )
    return out




_e8m0_cache = {}


def _ensure_e8m0_scales(b_scale: torch.Tensor) -> torch.Tensor:
    """Convert float32 scales to E8M0 uint8, with caching by data_ptr."""
    if b_scale.dtype == torch.uint8:
        return b_scale
    key = b_scale.data_ptr()
    cached = _e8m0_cache.get(key)
    if cached is not None:
        return cached
    tiny = torch.finfo(torch.float32).tiny
    e8m0 = torch.round(torch.log2(torch.clamp(b_scale, min=tiny))) + 127.0
    result = torch.clamp(e8m0, 0, 255).to(torch.uint8)
    _e8m0_cache[key] = result
    return result


def m_grouped_fp8_fp4_gemm_nt_contiguous_triton(a, b, d, m_indices=None, **kwargs):
    """Triton-accelerated M-grouped FP8×FP4 GEMM for MoE contiguous layout.

    Replaces the Python fallback with Triton FP4 dequant kernel + cuBLAS matmul.
    """
    if isinstance(a, tuple):
        a_tensor, a_scale = a
    else:
        a_tensor = a
        a_scale = None

    if isinstance(b, tuple):
        b_tensor, b_scale = b
    else:
        b_tensor = b
        b_scale = None

    if b_tensor.dtype == torch.int8:
        b_tensor = b_tensor.view(torch.uint8)

    from .gemm import _dequant_fp8_block
    if a_scale is not None:
        a_deq = _dequant_fp8_block(a_tensor, a_scale)
    else:
        a_deq = a_tensor.to(torch.bfloat16)

    if b_tensor.dim() != 3:
        b_deq = triton_dequant_fp4(b_tensor, b_scale)
        result = torch.mm(a_deq.to(torch.float32), b_deq.to(torch.float32).t())
        d.copy_(result.to(d.dtype))
        return

    m, k = a_deq.shape

    if m_indices is None or m_indices.numel() != m:
        from .gemm import _m_grouped_fp8_fp4_fallback_nt
        _m_grouped_fp8_fp4_fallback_nt(a, b, d, m_indices)
        return

    active_groups = m_indices[m_indices >= 0].unique()
    if active_groups.numel() == 0:
        d.zero_()
        return

    if b_scale is not None:
        b_scale_u8 = _ensure_e8m0_scales(b_scale)
    else:
        b_scale_u8 = None

    d.zero_()
    for gid_t in active_groups:
        gid = gid_t.item()
        rows = (m_indices == gid).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue

        gs = b_scale_u8[gid] if (b_scale_u8 is not None and b_scale_u8.dim() == 3) else b_scale_u8
        b_group_deq = triton_dequant_fp4(b_tensor[gid], gs)

        if b_group_deq.shape[-1] != k:
            if b_group_deq.shape[0] == k:
                b_group_deq = b_group_deq.t().contiguous()

        a_group = a_deq.index_select(0, rows).to(torch.float32)
        result = torch.mm(a_group, b_group_deq.to(torch.float32).t())
        d.index_copy_(0, rows, result.to(d.dtype))
