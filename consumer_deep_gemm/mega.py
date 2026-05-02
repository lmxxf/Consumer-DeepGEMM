"""DeepSeek V4 MegaMoE compatibility entry points.

These symbols keep vLLM's DeepSeek V4 import path alive while Consumer-DeepGEMM
is still bringing up the SM120 kernels. Weight transforms that are pure tensor
layout work are implemented; the distributed fused MegaMoE kernel is not.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _interleave_l1_weights(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    def interleave(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(
            torch.stack([gate, up], dim=2).reshape(g, n, *rest)
        )

    return interleave(l1_weights[0]), interleave(l1_weights[1])


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    if mn % 128 != 0:
        return sf.contiguous()
    result = (
        sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
        .transpose(2, 3)
        .reshape(num_groups, mn, packed_sf_k)
    )
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    l1_interleaved = _interleave_l1_weights(l1_weights)
    return (
        (l1_interleaved[0], _transpose_sf_for_utccp(l1_interleaved[1])),
        (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1])),
    )


def get_symm_buffer_for_mega_moe(*args, **kwargs):
    raise NotImplementedError(
        "Consumer-DeepGEMM has not implemented the distributed SM120 MegaMoE "
        "symmetric buffer path yet. Use the regular grouped GEMM path or keep "
        "vLLM's current non-DeepGEMM MoE backend until the native kernel lands."
    )


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
):
    raise NotImplementedError(
        "Consumer-DeepGEMM has not implemented fp8_fp4_mega_moe for SM120 yet."
    )
