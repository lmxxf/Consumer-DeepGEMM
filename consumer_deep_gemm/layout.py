"""Scale factor layout transforms matching DeepGEMM's API."""

import torch


def transform_sf_into_required_layout(sf: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Transform scale factor tensor into the layout required by the kernel.

    vLLM calls this through at least two DeepGEMM-compatible signatures:

    - ``(sf, mn=..., k=..., recipe=..., num_groups=..., is_sfa=...)``
    - ``(sf, m, n, recipe, num_groups)``

    Consumer-DeepGEMM's current Python fallbacks and native grouped FP4 path
    accept the original contiguous scale tensor, so this API is intentionally a
    compatibility no-op.
    """
    _ = args, kwargs
    return sf


def get_mn_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Return a TMA-aligned view. For SM120, just ensure contiguity."""
    if x.is_contiguous():
        return x
    return x.contiguous()


_mk_alignment_for_contiguous_layout = 128


def get_mk_alignment_for_contiguous_layout() -> int:
    """Return MK alignment requirements for contiguous grouped GEMM layout."""
    return _mk_alignment_for_contiguous_layout


def set_mk_alignment_for_contiguous_layout(alignment: int) -> None:
    global _mk_alignment_for_contiguous_layout
    _mk_alignment_for_contiguous_layout = int(alignment)


def get_theoretical_mk_alignment_for_contiguous_layout(*args, **kwargs) -> int:
    return 1
