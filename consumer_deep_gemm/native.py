"""Native CUDA extension loader.

The package must remain importable without compiling CUDA. Set
CONSUMER_DEEP_GEMM_BUILD_CUDA=1 during installation to build the CUTLASS-backed
extension.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _load_native():
    try:
        from . import _C  # type: ignore
    except ImportError:
        return None
    return _C


def is_available() -> bool:
    return _load_native() is not None


def build_info() -> dict[str, object]:
    ext = _load_native()
    if ext is None:
        return {
            "available": False,
            "cutlass_sm120_probe": False,
            "arch": None,
        }
    return {
        "available": bool(ext.is_available()),
        "cutlass_sm120_probe": bool(ext.cutlass_sm120_probe_compiled()),
        "arch": ext.cutlass_sm120_probe_arch(),
    }


def m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs):
    ext = _load_native()
    if ext is None or not hasattr(ext, "m_grouped_fp8_fp4_gemm_nt_contiguous"):
        return None
    return ext.m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs)
