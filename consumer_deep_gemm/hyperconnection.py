"""Hyper-Connection (mHC) kernel: TF32 pre-norm GEMM with split-K and square-sum output.

Matches DeepGEMM's tf32_hc_prenorm_gemm API.
"""

import torch
from typing import Optional


def tf32_hc_prenorm_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    sqr_sum: torch.Tensor,
    num_splits: Optional[int] = None,
) -> None:
    """
    D[num_splits, M, N] = A[M, K] @ B[K, N]^T  (per-split)
    sqr_sum[num_splits, M] = sum(D[..., :]^2, dim=-1)

    A is BF16, B is FP32, D is FP32, sqr_sum is FP32.
    """
    n_splits = num_splits if num_splits is not None else 1
    M, K = a.shape
    N = b.shape[0]

    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)

    gemm_out = torch.mm(a_f, b_f.t())

    if n_splits > 1:
        split_n = N // n_splits
        for s in range(n_splits):
            start = s * split_n
            end = start + split_n
            d[s].copy_(gemm_out[:, start:end])
            sqr_sum[s].copy_(gemm_out[:, start:end].pow(2).sum(dim=-1))
    else:
        d.copy_(gemm_out)
        sqr_sum.copy_(gemm_out.pow(2).sum(dim=-1))
