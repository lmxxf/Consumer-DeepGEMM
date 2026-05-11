"""Triton FP4 dequant + grouped MoE GEMM for SM120/SM121.

Replaces the Python fallback in gemm.py with Triton kernels that
eliminate per-expert Python loop overhead and temporary tensor allocation.
"""

import os

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




@triton.jit
def _e2m1_to_fp8(idx):
    """Convert 4-bit E2M1 index to FP8 E4M3 bit pattern (lossless)."""
    sign = (idx >> 3) & 1
    mag = idx & 0x07
    e = tl.where(mag == 0, 0,
        tl.where(mag == 1, 6,
        tl.where(mag == 2, 7,
        tl.where(mag == 3, 7,
        tl.where(mag == 4, 8,
        tl.where(mag == 5, 8,
        tl.where(mag == 6, 9, 9)))))))
    m = tl.where(mag == 0, 0,
        tl.where(mag == 1, 0,
        tl.where(mag == 2, 0,
        tl.where(mag == 3, 4,
        tl.where(mag == 4, 0,
        tl.where(mag == 5, 4,
        tl.where(mag == 6, 0, 4)))))))
    return ((sign << 7) | (e << 3) | m).to(tl.uint8)


@triton.jit
def _dequant_fp4_vals(packed, scale_f):
    """Inline helper: unpack uint8 -> two float32 values, apply scale."""
    low_idx = (packed & 0x0F).to(tl.int32)
    high_idx = ((packed >> 4) & 0x0F).to(tl.int32)

    low_sign = (low_idx >> 3).to(tl.float32)
    low_mag = low_idx & 0x07
    low_abs = tl.where(low_mag == 0, 0.0,
              tl.where(low_mag == 1, 0.5,
              tl.where(low_mag == 2, 1.0,
              tl.where(low_mag == 3, 1.5,
              tl.where(low_mag == 4, 2.0,
              tl.where(low_mag == 5, 3.0,
              tl.where(low_mag == 6, 4.0, 6.0)))))))
    low_val = tl.where(low_sign > 0.5, -low_abs, low_abs) * scale_f

    high_sign = (high_idx >> 3).to(tl.float32)
    high_mag = high_idx & 0x07
    high_abs = tl.where(high_mag == 0, 0.0,
               tl.where(high_mag == 1, 0.5,
               tl.where(high_mag == 2, 1.0,
               tl.where(high_mag == 3, 1.5,
               tl.where(high_mag == 4, 2.0,
               tl.where(high_mag == 5, 3.0,
               tl.where(high_mag == 6, 4.0, 6.0)))))))
    high_val = tl.where(high_sign > 0.5, -high_abs, high_abs) * scale_f

    return low_val, high_val


@triton.jit
def _fused_fp8_matmul_kernel(
    a_ptr, a_stride_m, a_stride_k,
    b_ptr, b_stride_n, b_stride_k,
    bs_ptr, bs_stride_n, bs_stride_k,
    d_ptr, d_stride_m, d_stride_n,
    M, N, K: tl.constexpr,
    K_PACKED: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # unused, kept for API compat
):
    """Fused FP4->FP8 dequant + FP8 tensor core matmul.

    Iterates in scale-block-sized chunks (32 K values = 16 packed bytes).
    Each chunk: unpack 16 bytes -> 32 FP8 values, load 32 A columns as FP8,
    tl.dot(fp8[M,32], fp8[N,32]^T) with exact per-block E8M0 scale.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 32 K values = 16 packed bytes = 1 scale block
    CHUNK_K: tl.constexpr = 32
    CHUNK_K_PACKED: tl.constexpr = 16

    for k_chunk in range(0, K, CHUNK_K):
        k_packed_start = k_chunk // 2
        k_packed_offs = k_packed_start + tl.arange(0, CHUNK_K_PACKED)
        k_packed_mask = k_packed_offs < K_PACKED

        # Load B packed: [BLOCK_N, 16] uint8
        b_packed = tl.load(
            b_ptr + n_offs[:, None] * b_stride_n + k_packed_offs[None, :] * b_stride_k,
            mask=n_mask[:, None] & k_packed_mask[None, :],
            other=0
        ).to(tl.uint8)

        # Unpack to FP8: low/high nibbles -> [BLOCK_N, 16] each
        low_fp8 = _e2m1_to_fp8((b_packed & 0x0F).to(tl.int32))
        high_fp8 = _e2m1_to_fp8(((b_packed >> 4) & 0x0F).to(tl.int32))
        low_f8 = low_fp8.to(tl.float8e4nv, bitcast=True)
        high_f8 = high_fp8.to(tl.float8e4nv, bitcast=True)

        # Load A even/odd columns: [BLOCK_M, 16] each
        k_even = k_chunk + tl.arange(0, CHUNK_K_PACKED) * 2
        k_odd = k_chunk + tl.arange(0, CHUNK_K_PACKED) * 2 + 1

        a_even = tl.load(
            a_ptr + m_offs[:, None] * a_stride_m + k_even[None, :] * a_stride_k,
            mask=m_mask[:, None] & (k_even[None, :] < K), other=0.0
        ).to(tl.float8e4nv)

        a_odd = tl.load(
            a_ptr + m_offs[:, None] * a_stride_m + k_odd[None, :] * a_stride_k,
            mask=m_mask[:, None] & (k_odd[None, :] < K), other=0.0
        ).to(tl.float8e4nv)

        # FP8 dot: two K=16 halves (even/odd) — but K=16 < 32 minimum for tl.dot.
        # Concat even+odd to make K=32: A_cat[M,32] = [a_even, a_odd]
        #                                 B_cat[N,32] = [low_f8, high_f8]
        # Then D = A_cat @ B_cat^T = a_even@low^T + a_odd@high^T (correct!)
        # because the cross terms (a_even@high^T, a_odd@low^T) are between
        # independent K ranges so they correctly contribute zero in the sum.
        # Wait — NO. Concat makes them share the same K dimension, so the dot
        # DOES compute cross terms. That's wrong.
        #
        # Actually: D[m,n] = sum_{j=0}^{31} A_cat[m,j] * B_cat[n,j]
        #         = sum_{j=0}^{15} a_even[m,j]*low[n,j] + sum_{j=16}^{31} a_odd[m,j-16]*high[n,j-16]
        #         = a_even @ low^T + a_odd @ high^T
        # This IS correct! Concatenation along K works because the two halves
        # occupy different j-positions, there are no cross terms.

        a_cat = tl.join(a_even, a_odd).reshape(BLOCK_M, CHUNK_K)       # [M, 32]
        b_cat = tl.join(low_f8, high_f8).reshape(BLOCK_N, CHUNK_K)     # [N, 32]

        dot_result = tl.dot(a_cat, tl.trans(b_cat))  # [BLOCK_M, BLOCK_N], K=32 ✓

        # Exact per-block E8M0 scale (1 scale per N per chunk)
        scale_idx = k_packed_start // 16
        s = tl.load(
            bs_ptr + n_offs * bs_stride_n + scale_idx * bs_stride_k,
            mask=n_mask & (scale_idx < K_SCALE), other=127
        ).to(tl.float32)
        scale_f = tl.exp2(s - 127.0)  # [BLOCK_N]

        acc += dot_result * scale_f[None, :]

    d_block = acc.to(tl.bfloat16)
    tl.store(
        d_ptr + m_offs[:, None] * d_stride_m + n_offs[None, :] * d_stride_n,
        d_block,
        mask=m_mask[:, None] & n_mask[None, :]
    )


@triton.jit
def _fused_dequant_matmul_kernel(
    # A: [M, K] bf16 (already FP8-dequanted)
    a_ptr, a_stride_m, a_stride_k,
    # B: [N, K_packed] uint8 packed FP4
    b_ptr, b_stride_n, b_stride_k,
    # B scale: [N, K_scale] uint8 E8M0
    bs_ptr, bs_stride_n, bs_stride_k,
    # D: [M, N] bf16 output
    d_ptr, d_stride_m, d_stride_n,
    M, N, K: tl.constexpr,
    K_PACKED: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # number of actual K values per iteration (must be even)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2

    for k_start in range(0, K, BLOCK_K):
        k_packed_start = k_start // 2
        k_packed_offs = k_packed_start + tl.arange(0, BLOCK_K_PACKED)
        k_packed_mask = k_packed_offs < K_PACKED

        # Load A even columns: a[m, k_start], a[m, k_start+2], ...
        # These correspond to the "low" nibble of each packed byte
        k_even_offs = k_start + tl.arange(0, BLOCK_K_PACKED) * 2
        a_even = tl.load(
            a_ptr + m_offs[:, None] * a_stride_m + k_even_offs[None, :] * a_stride_k,
            mask=m_mask[:, None] & (k_even_offs[None, :] < K),
            other=0.0
        ).to(tl.float32)

        # Load A odd columns: a[m, k_start+1], a[m, k_start+3], ...
        k_odd_offs = k_start + tl.arange(0, BLOCK_K_PACKED) * 2 + 1
        a_odd = tl.load(
            a_ptr + m_offs[:, None] * a_stride_m + k_odd_offs[None, :] * a_stride_k,
            mask=m_mask[:, None] & (k_odd_offs[None, :] < K),
            other=0.0
        ).to(tl.float32)

        # Load B packed block: [BLOCK_N, BLOCK_K_PACKED]
        b_packed = tl.load(
            b_ptr + n_offs[:, None] * b_stride_n + k_packed_offs[None, :] * b_stride_k,
            mask=n_mask[:, None] & k_packed_mask[None, :],
            other=0
        ).to(tl.uint8)

        # Load scales: each covers 16 packed bytes = 32 values
        scale_k_offs = k_packed_offs // 16
        b_scale = tl.load(
            bs_ptr + n_offs[:, None] * bs_stride_n + scale_k_offs[None, :] * bs_stride_k,
            mask=n_mask[:, None] & (scale_k_offs[None, :] < K_SCALE),
            other=127
        ).to(tl.float32)
        scale_f = tl.exp2(b_scale - 127.0)

        low_val, high_val = _dequant_fp4_vals(b_packed, scale_f)

        # [BLOCK_M, BLOCK_K_PACKED] x [BLOCK_N, BLOCK_K_PACKED]^T = [BLOCK_M, BLOCK_N]
        acc += tl.dot(a_even, tl.trans(low_val), input_precision="ieee")
        acc += tl.dot(a_odd, tl.trans(high_val), input_precision="ieee")

    # Store result
    d_block = acc.to(tl.bfloat16)
    tl.store(
        d_ptr + m_offs[:, None] * d_stride_m + n_offs[None, :] * d_stride_n,
        d_block,
        mask=m_mask[:, None] & n_mask[None, :]
    )


@triton.jit
def _dot_scaled_matmul_kernel(
    # A: [M, K] bf16 activation (will be cast to FP8 e4m3 for dot_scaled)
    a_ptr, a_stride_m, a_stride_k,
    # B: [N, K_packed] uint8 packed FP4 e2m1 weight (will be transposed internally)
    b_ptr, b_stride_n, b_stride_k,
    # B scale: [N, K_scale] uint8 E8M0
    bs_ptr, bs_stride_n, bs_stride_k,
    # D: [M, N] bf16 output
    d_ptr, d_stride_m, d_stride_n,
    M, N, K: tl.constexpr,
    K_PACKED: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # unused, API compat
):
    """dot_scaled FP8×FP4 matmul: A_bf16 cast to e4m3, B stays packed e2m1.

    Iterates in 32-K-value chunks (= 32 FP8 bytes + 16 packed FP4 bytes = 1 scale block).
    Each chunk: tl.dot_scaled(a_e4m3[M,32], b_e2m1[16,N], ...) with exact E8M0 scale.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    CHUNK_K: tl.constexpr = 32
    CHUNK_K_PACKED: tl.constexpr = 16

    for k_start in range(0, K, CHUNK_K):
        k_offs = k_start + tl.arange(0, CHUNK_K)
        kp_start = k_start // 2
        kp_offs = kp_start + tl.arange(0, CHUNK_K_PACKED)

        # A: [BLOCK_M, 32] bf16 -> cast to uint8 (FP8 e4m3 bits)
        a_bf16 = tl.load(
            a_ptr + m_offs[:, None] * a_stride_m + k_offs[None, :] * a_stride_k,
            mask=m_mask[:, None] & (k_offs[None, :] < K), other=0.0
        )
        a_fp8 = a_bf16.to(tl.float8e4nv)

        # B: [CHUNK_K_PACKED, BLOCK_N] uint8 packed e2m1, K-major
        # We need B in [K_packed, N] layout for dot_scaled rhs
        b_chunk = tl.load(
            b_ptr + n_offs[None, :] * b_stride_n + kp_offs[:, None] * b_stride_k,
            mask=(kp_offs[:, None] < K_PACKED) & n_mask[None, :], other=0
        )

        # Scales: 1 per row/col per chunk
        scale_idx = k_start // 32

        # A scale: [BLOCK_M, 1] — use neutral scale 127 (= 2^0 = 1.0) since A is already in real values
        # dot_scaled will multiply by 2^(scale-127), so 127 means no scaling
        a_sc = tl.full((BLOCK_M, 1), 127, dtype=tl.uint8)

        # B scale: [BLOCK_N, 1]
        b_sc_val = tl.load(
            bs_ptr + n_offs * bs_stride_n + scale_idx * bs_stride_k,
            mask=n_mask & (scale_idx < K_SCALE), other=127
        )
        b_sc = b_sc_val[:, None]

        acc += tl.dot_scaled(a_fp8, a_sc, "e4m3", b_chunk, b_sc, "e2m1")

    tl.store(
        d_ptr + m_offs[:, None] * d_stride_m + n_offs[None, :] * d_stride_n,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :]
    )


@triton.jit
def _grouped_dot_scaled_kernel(
    # A_sorted: [M_total, K] bf16 (all experts concatenated, sorted by expert_id)
    a_ptr, a_stride_m, a_stride_k,
    # B: [G, N, K_packed] uint8 packed FP4 (all experts)
    b_ptr, b_stride_g, b_stride_n, b_stride_k,
    # B scale: [G, N, K_scale] uint8 E8M0
    bs_ptr, bs_stride_g, bs_stride_n, bs_stride_k,
    # D_sorted: [M_total, N] bf16 output
    d_ptr, d_stride_m, d_stride_n,
    # Segment metadata
    seg_expert_ids_ptr,  # [n_segs] int32: expert id for each segment
    seg_starts_ptr,      # [n_segs] int32: start row in A_sorted for each segment
    seg_counts_ptr,      # [n_segs] int32: number of rows in each segment
    N, K: tl.constexpr,
    K_PACKED: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Grouped dot_scaled: one kernel launch processes all experts.

    Grid: (ceil(max_M/BLOCK_M), ceil(N/BLOCK_N), n_segs)
    program_id(2) = segment index (which expert)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    seg_id = tl.program_id(2)

    # Load segment metadata
    expert_id = tl.load(seg_expert_ids_ptr + seg_id)
    seg_start = tl.load(seg_starts_ptr + seg_id)
    seg_count = tl.load(seg_counts_ptr + seg_id)

    # Skip if this M-block is beyond this segment's rows
    m_base = pid_m * BLOCK_M
    if m_base >= seg_count:
        return

    # Global row offsets in A_sorted / D_sorted
    m_offs = seg_start + m_base + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = (m_base + tl.arange(0, BLOCK_M)) < seg_count
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    CHUNK_K: tl.constexpr = 32
    CHUNK_K_PACKED: tl.constexpr = 16

    for k_start in range(0, K, CHUNK_K):
        k_offs = k_start + tl.arange(0, CHUNK_K)
        kp_start = k_start // 2
        kp_offs = kp_start + tl.arange(0, CHUNK_K_PACKED)

        # A: [BLOCK_M, 32] bf16 from A_sorted
        a_bf16 = tl.load(
            a_ptr + m_offs[:, None] * a_stride_m + k_offs[None, :] * a_stride_k,
            mask=m_mask[:, None] & (k_offs[None, :] < K), other=0.0
        )
        a_fp8 = a_bf16.to(tl.float8e4nv)

        # B: [CHUNK_K_PACKED, BLOCK_N] from B[expert_id]
        b_chunk = tl.load(
            b_ptr + expert_id * b_stride_g + n_offs[None, :] * b_stride_n + kp_offs[:, None] * b_stride_k,
            mask=(kp_offs[:, None] < K_PACKED) & n_mask[None, :], other=0
        )

        # Scales
        scale_idx = k_start // 32
        a_sc = tl.full((BLOCK_M, 1), 127, dtype=tl.uint8)
        b_sc_val = tl.load(
            bs_ptr + expert_id * bs_stride_g + n_offs * bs_stride_n + scale_idx * bs_stride_k,
            mask=n_mask & (scale_idx < K_SCALE), other=127
        )
        b_sc = b_sc_val[:, None]

        acc += tl.dot_scaled(a_fp8, a_sc, "e4m3", b_chunk, b_sc, "e2m1")

    # Store to D_sorted
    tl.store(
        d_ptr + m_offs[:, None] * d_stride_m + n_offs[None, :] * d_stride_n,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :]
    )


def triton_fused_fp4_matmul_nt(a_bf16, b_packed, b_scale_u8, out, use_fp8=True):
    """Fused FP4 dequant + matmul: D = A @ dequant(B)^T.

    A: [M, K] bf16
    B: [N, K//2] uint8 packed FP4
    b_scale: [N, K//32] uint8 E8M0
    out: [M, N] bf16
    use_fp8: if True, use dot_scaled path (fastest); else use float32 path (exact)
    """
    M, K = a_bf16.shape
    N = b_packed.shape[0]
    K_packed = K // 2
    K_scale = K // 32

    BLOCK_M = min(64, triton.next_power_of_2(M))
    BLOCK_N = 32
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    kernel = _dot_scaled_matmul_kernel if use_fp8 else _fused_dequant_matmul_kernel

    kernel[grid](
        a_bf16, a_bf16.stride(0), a_bf16.stride(1),
        b_packed, b_packed.stride(0), b_packed.stride(1),
        b_scale_u8, b_scale_u8.stride(0), b_scale_u8.stride(1),
        out, out.stride(0), out.stride(1),
        M, N, K, K_packed, K_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


def triton_grouped_fused_fp4_matmul_nt(
    a_sorted: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale_u8: torch.Tensor,
    d_sorted: torch.Tensor,
    seg_expert_ids: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_counts: torch.Tensor,
):
    """One-launch grouped FP8/BF16 x FP4 MoE matmul for sorted rows.

    ``a_sorted`` and ``d_sorted`` contain only valid rows, sorted by expert id.
    Segment tensors describe each contiguous expert slice.
    """
    if a_sorted.numel() == 0 or seg_expert_ids.numel() == 0:
        return

    _, k = a_sorted.shape
    n = b_packed.shape[1]
    k_packed = k // 2
    k_scale = k // 32
    n_segs = int(seg_expert_ids.numel())
    max_seg_m = int(seg_counts.max().item())

    block_m = min(64, triton.next_power_of_2(max_seg_m))
    block_n = 32
    grid = (triton.cdiv(max_seg_m, block_m), triton.cdiv(n, block_n), n_segs)

    _grouped_dot_scaled_kernel[grid](
        a_sorted, a_sorted.stride(0), a_sorted.stride(1),
        b_packed, b_packed.stride(0), b_packed.stride(1), b_packed.stride(2),
        b_scale_u8, b_scale_u8.stride(0), b_scale_u8.stride(1), b_scale_u8.stride(2),
        d_sorted, d_sorted.stride(0), d_sorted.stride(1),
        seg_expert_ids, seg_starts, seg_counts,
        n, k, k_packed, k_scale,
        BLOCK_M=block_m, BLOCK_N=block_n,
    )


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
    """Triton-accelerated M-grouped FP8×FP4 GEMM for MoE contiguous layout."""
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

    # Skip FP8 dequant if activation is already BF16/FP32
    if a_tensor.dtype in (torch.bfloat16, torch.float32, torch.float16):
        a_deq = a_tensor.to(torch.bfloat16)
    elif a_scale is not None:
        from .gemm import _dequant_fp8_block
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

    if b_scale is not None:
        b_scale_u8 = _ensure_e8m0_scales(b_scale)
    else:
        b_scale_u8 = None

    # Sort rows by expert ID — all GPU ops, no CPU sync
    sort_indices = torch.argsort(m_indices)
    sorted_indices = m_indices[sort_indices]

    # Find where each expert's rows start/end using pure GPU ops
    # diff[i] = 1 where expert changes, then cumsum to get group boundaries
    valid_mask = sorted_indices >= 0
    n_valid = valid_mask.sum()  # this is a GPU tensor, no sync yet

    n_valid_int = int(n_valid.item())  # single sync, unavoidable
    if n_valid_int == 0:
        d.zero_()
        return

    # Extract valid rows (skip -1 padding)
    valid_sort_indices = sort_indices[valid_mask]
    valid_expert_ids = sorted_indices[valid_mask]

    # Find segment boundaries: where expert ID changes
    changes = torch.zeros(n_valid_int, dtype=torch.bool, device=d.device)
    changes[0] = True
    if n_valid_int > 1:
        changes[1:] = valid_expert_ids[1:] != valid_expert_ids[:-1]
    seg_starts = changes.nonzero(as_tuple=False).flatten()

    # Gather all sorted A rows at once (one index_select, not 6)
    a_sorted = a_deq.index_select(0, valid_sort_indices)

    d.zero_()
    N_out = b_tensor.shape[1]
    d_sorted = torch.empty(n_valid_int, N_out, dtype=d.dtype, device=d.device)

    use_grouped_launch = os.getenv("CDG_GROUPED_DOT_SCALED", "0") == "1"

    if use_grouped_launch and b_scale_u8 is not None and b_scale_u8.dim() == 3:
        seg_ends = torch.empty_like(seg_starts)
        if seg_starts.numel() > 1:
            seg_ends[:-1] = seg_starts[1:]
        seg_ends[-1] = n_valid_int
        seg_counts = (seg_ends - seg_starts).to(torch.int32)
        seg_expert_ids = valid_expert_ids[seg_starts].to(torch.int32)
        triton_grouped_fused_fp4_matmul_nt(
            a_sorted,
            b_tensor,
            b_scale_u8,
            d_sorted,
            seg_expert_ids,
            seg_starts.to(torch.int32),
            seg_counts,
        )
    else:
        # Default path. A one-launch grouped kernel is available behind
        # CDG_GROUPED_DOT_SCALED=1, but vLLM end-to-end testing showed it is
        # slower than per-expert launches despite a standalone microbenchmark
        # win, because metadata/padding/3D-grid costs dominate in service.
        seg_starts_cpu = seg_starts.cpu()
        expert_ids_at_starts = valid_expert_ids[seg_starts].cpu()
        n_segs = seg_starts_cpu.numel()
        for i in range(n_segs):
            gid = expert_ids_at_starts[i].item()
            start = seg_starts_cpu[i].item()
            end = seg_starts_cpu[i + 1].item() if i + 1 < n_segs else n_valid_int
            gs = b_scale_u8[gid] if (b_scale_u8 is not None and b_scale_u8.dim() == 3) else b_scale_u8
            triton_fused_fp4_matmul_nt(
                a_sorted[start:end], b_tensor[gid], gs, d_sorted[start:end])

    # Scatter results back in one op
    d.index_copy_(0, valid_sort_indices, d_sorted)
