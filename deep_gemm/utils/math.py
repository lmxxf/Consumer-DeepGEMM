"""Minimal math helpers used by vLLM tests and wrappers."""

from __future__ import annotations

import torch


def ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def align(x: int, alignment: int) -> int:
    alignment = int(alignment)
    return ceil_div(int(x), alignment) * alignment


def per_token_cast_to_fp4(*args, **kwargs):
    raise NotImplementedError("Consumer-DeepGEMM does not expose FP4 quantization helpers yet")


def per_block_cast_to_fp8(x: torch.Tensor, block_k: int = 128, *args, **kwargs):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    M, K = x.shape[0], x.shape[1]
    n_blocks = ceil_div(K, block_k)
    pad = n_blocks * block_k - K
    if pad:
        x_padded = torch.nn.functional.pad(x, (0, pad))
    else:
        x_padded = x
    x_blocks = x_padded.reshape(M, n_blocks, block_k)
    amax = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / fp8_max).squeeze(-1)
    x_scaled = x_blocks / amax * fp8_max
    x_fp8 = x_scaled.reshape(M, n_blocks * block_k)
    if pad:
        x_fp8 = x_fp8[:, :K]
    return x_fp8.to(torch.float8_e4m3fn), scale


def per_token_cast_to_fp8(x: torch.Tensor, *args, **kwargs):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / fp8_max
    x_scaled = x / amax * fp8_max
    return x_scaled.to(torch.float8_e4m3fn), scale
