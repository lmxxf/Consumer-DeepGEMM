"""
Consumer-DeepGEMM: DeepGEMM-compatible API for consumer Blackwell GPUs (SM120/SM121).

Drop-in replacement: `import consumer_deep_gemm as deep_gemm`
"""

import os
import time
import torch

from .gemm import (
    fp8_gemm_nt, fp8_gemm_nn, fp8_gemm_tn, fp8_gemm_tt,
    fp8_fp4_gemm_nt, fp8_fp4_gemm_nn,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nn_contiguous,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_fp4_gemm_nn_contiguous,
    m_grouped_fp8_gemm_nt_masked,
    m_grouped_fp8_fp4_gemm_nt_masked,
    bf16_gemm_nt, bf16_gemm_nn, bf16_gemm_tn, bf16_gemm_tt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nn_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    cublaslt_gemm_nt,
)
from .attention import (
    fp8_fp4_mqa_logits,
    get_paged_mqa_logits_metadata,
    fp8_fp4_paged_mqa_logits,
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
)
from .hyperconnection import tf32_hc_prenorm_gemm
from .einsum import einsum, fp8_einsum
from .layout import (
    transform_sf_into_required_layout,
    get_mn_major_tma_aligned_tensor,
    get_mk_alignment_for_contiguous_layout,
    set_mk_alignment_for_contiguous_layout,
    get_theoretical_mk_alignment_for_contiguous_layout,
)
from .utils import set_num_sms, get_num_sms, set_tc_util, get_tc_util
from .native import build_info as native_build_info, is_available as native_is_available
from .mega import (
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
)

# Legacy aliases
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked

__version__ = '0.1.0'


# ============================================================
# Profiling: set CDG_PROFILE=1 to enable, CDG_PROFILE_INTERVAL=N
# to print summary every N calls (default 100)
# ============================================================

if os.environ.get("CDG_PROFILE", "0") == "1":
    import functools
    from collections import defaultdict

    _prof_stats = defaultdict(lambda: [0, 0.0])  # {name: [count, total_seconds]}
    _prof_call_count = 0
    _prof_interval = int(os.environ.get("CDG_PROFILE_INTERVAL", "10"))

    def _prof_wrap(fn, name):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            global _prof_call_count
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            _prof_stats[name][0] += 1
            _prof_stats[name][1] += elapsed
            _prof_call_count += 1
            if _prof_call_count % _prof_interval == 0:
                print(f"\n=== CDG Profile ({_prof_call_count} calls) ===")
                for n, (c, t) in sorted(_prof_stats.items(), key=lambda x: -x[1][1]):
                    if c > 0:
                        print(f"  {n:50s}  calls={c:6d}  total={t:8.2f}s  avg={t/c*1000:8.2f}ms")
                print(f"  {'TOTAL':50s}  calls={sum(v[0] for v in _prof_stats.values()):6d}  total={sum(v[1] for v in _prof_stats.values()):8.2f}s")
                print()
            return result
        return wrapper

    _PROFILE_TARGETS = [
        "fp8_gemm_nt", "fp8_gemm_nn", "fp8_gemm_tn", "fp8_gemm_tt",
        "fp8_fp4_gemm_nt", "fp8_fp4_gemm_nn",
        "m_grouped_fp8_gemm_nt_contiguous", "m_grouped_fp8_gemm_nn_contiguous",
        "m_grouped_fp8_fp4_gemm_nt_contiguous", "m_grouped_fp8_fp4_gemm_nn_contiguous",
        "m_grouped_fp8_gemm_nt_masked", "m_grouped_fp8_fp4_gemm_nt_masked",
        "bf16_gemm_nt", "bf16_gemm_nn", "bf16_gemm_tn", "bf16_gemm_tt",
        "m_grouped_bf16_gemm_nt_contiguous", "m_grouped_bf16_gemm_nn_contiguous",
        "m_grouped_bf16_gemm_nt_masked",
        "cublaslt_gemm_nt",
        "fp8_fp4_mqa_logits", "get_paged_mqa_logits_metadata",
        "fp8_fp4_paged_mqa_logits", "fp8_mqa_logits", "fp8_paged_mqa_logits",
        "tf32_hc_prenorm_gemm",
        "einsum", "fp8_einsum",
    ]

    import sys
    _this = sys.modules[__name__]
    for _name in _PROFILE_TARGETS:
        _fn = getattr(_this, _name, None)
        if _fn is not None and callable(_fn):
            setattr(_this, _name, _prof_wrap(_fn, _name))
    del _this, _name, _fn
