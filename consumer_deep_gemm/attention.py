"""MQA/MLA attention kernels matching DeepGEMM's API.

Phase 1: Pure PyTorch fallback
Phase 2: CUTLASS SM120 MLA kernel (Example 77)
"""

import torch
from typing import Optional, Tuple


def fp8_fp4_mqa_logits(
    q,
    k: torch.Tensor,
    logits: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    **kwargs,
) -> None:
    """Multi-query attention logits: logits = Q @ K^T * scale."""
    if isinstance(q, tuple):
        q_tensor, q_scale = q
        q_f = q_tensor.to(torch.float32) * q_scale.to(torch.float32)
    else:
        q_f = q.to(torch.float32)

    k_f = k.to(torch.float32)
    if k_scale is not None:
        k_f = k_f * k_scale.to(torch.float32)

    result = torch.matmul(q_f, k_f.transpose(-2, -1))
    logits.copy_(result)


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build scheduling metadata for paged MQA logits.

    Distributes KV-cache segments across SMs. Output shape: [num_sms + 1, 2],
    dtype int32, on the same device as context_lens.
    Each row is (q_atom_idx, kv_split_idx).
    """
    assert context_lens.dim() == 2
    batch_size = context_lens.size(0)
    next_n = context_lens.size(1)
    ctx_cpu = context_lens.to("cpu", non_blocking=False).to(torch.int32)

    next_n_atom = 2 if next_n >= 2 else 1
    num_next_n_atoms = (next_n + next_n_atom - 1) // next_n_atom

    last_col = ctx_cpu[:, -1]
    num_segs = ((last_col.to(torch.int64) + block_kv - 1) // block_kv).tolist()
    prefix = []
    s = 0
    for ns in num_segs:
        s += int(ns)
        prefix.append(s)

    total = s * num_next_n_atoms
    schedule = torch.empty((num_sms + 1, 2), dtype=torch.int32)

    q_div = total // num_sms
    r = total % num_sms
    for sm_idx in range(num_sms + 1):
        seg_start = sm_idx * q_div + min(sm_idx, r)
        lo, hi = 0, batch_size
        while lo < hi:
            mid = (lo + hi) // 2
            if prefix[mid] * num_next_n_atoms <= seg_start:
                lo = mid + 1
            else:
                hi = mid
        q_idx = lo
        if q_idx == 0:
            offset_in_q = seg_start
        else:
            offset_in_q = seg_start - prefix[q_idx - 1] * num_next_n_atoms
        num_segs_q = prefix[q_idx] - (prefix[q_idx - 1] if q_idx > 0 else 0) if q_idx < batch_size else 0
        if num_segs_q > 0:
            atom_idx = offset_in_q // num_segs_q
            kv_split_idx = offset_in_q % num_segs_q
        else:
            atom_idx = 0
            kv_split_idx = 0
        q_atom_idx = q_idx * num_next_n_atoms + atom_idx
        schedule[sm_idx, 0] = q_atom_idx
        schedule[sm_idx, 1] = kv_split_idx

    return schedule.to(device=context_lens.device)


def fp8_fp4_paged_mqa_logits(
    q,
    k_cache: torch.Tensor,
    logits: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    **kwargs,
) -> None:
    """Paged MQA logits with KV cache block table."""
    fp8_fp4_mqa_logits(q, k_cache, logits, k_scale=k_scale, **kwargs)


# Legacy aliases
def fp8_mqa_logits(q, k, logits, **kwargs):
    fp8_fp4_mqa_logits(q, k, logits, **kwargs)


def fp8_paged_mqa_logits(q, k_cache, logits, **kwargs):
    fp8_fp4_paged_mqa_logits(q, k_cache, logits, **kwargs)
