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


def get_paged_mqa_logits_metadata(*args, **kwargs):
    """Metadata for paged MQA logits. Returns None for fallback path."""
    return None


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
