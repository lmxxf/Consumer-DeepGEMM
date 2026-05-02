"""
Consumer-DeepGEMM: DeepGEMM-compatible API for consumer Blackwell GPUs (SM120/SM121).

Drop-in replacement: `import consumer_deep_gemm as deep_gemm`
"""

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
)
from .utils import set_num_sms, get_num_sms, set_tc_util, get_tc_util

# Legacy aliases
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked

__version__ = '0.1.0'
