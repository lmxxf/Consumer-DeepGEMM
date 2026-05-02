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
        "cutlass_mxfp8_mxfp4_probe": bool(ext.cutlass_mxfp8_mxfp4_probe_compiled()),
        "arch": ext.cutlass_sm120_probe_arch(),
    }


def cutlass_mxfp8_mxfp4_can_implement_probe(a, b, d) -> bool | None:
    ext = _load_native()
    if ext is None or not hasattr(ext, "cutlass_mxfp8_mxfp4_can_implement_probe"):
        return None
    if not (getattr(a, "is_cuda", False) and getattr(b, "is_cuda", False) and getattr(d, "is_cuda", False)):
        return None
    return bool(ext.cutlass_mxfp8_mxfp4_can_implement_probe(a, b, d))


def _first_tensor(value):
    if isinstance(value, tuple):
        return value[0]
    return value


def m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs):
    ext = _load_native()
    if ext is None or not hasattr(ext, "m_grouped_fp8_fp4_gemm_nt_contiguous"):
        return None
    if len(args) < 3:
        return None
    a = _first_tensor(args[0])
    b = _first_tensor(args[1])
    d = args[2]
    if not (getattr(a, "is_cuda", False) and getattr(b, "is_cuda", False) and getattr(d, "is_cuda", False)):
        return None
    return ext.m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs)
