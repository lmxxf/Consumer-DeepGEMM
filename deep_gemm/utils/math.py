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


def per_block_cast_to_fp8(x: torch.Tensor, *args, **kwargs):
    scale = torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)
    return x.to(torch.float8_e4m3fn), scale


def per_token_cast_to_fp8(x: torch.Tensor, *args, **kwargs):
    scale = torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)
    return x.to(torch.float8_e4m3fn), scale
