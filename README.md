# Consumer-DeepGEMM

DeepGEMM-compatible drop-in replacement for consumer-grade Blackwell GPUs (SM120/SM121).

Fixes the **silent computation error** in vLLM's Marlin FP4 MoE kernel on DGX Spark / RTX 5090 — where DeepSeek V4 Flash 280B produces garbage output for certain expert combinations.

## Why This Exists

[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) only supports data-center GPUs (SM90 Hopper, SM100 Blackwell). Consumer Blackwell (SM120/SM121) uses different MMA instructions (`mma.sync.aligned` + block-scaled MX instead of `tcgen05` + TMA). vLLM falls back to Marlin for FP4 MoE on SM120+, but Marlin **silently computes wrong results** on these GPUs ([vllm#40928](https://github.com/vllm-project/vllm/issues/40928)).

This project replaces Marlin's FP4 path with CUTLASS SM120 templates that produce correct results.

## Architecture

```
vLLM MoE dispatch → deep_gemm API (unchanged)
    ↓
Consumer-DeepGEMM (this project)
    ↓
CUTLASS SM120 grouped GEMM templates
    ↓
SM120/SM121 mma.sync.aligned + block-scaled MX hardware
```

## Target Hardware

| GPU | Compute Capability | Status |
|-----|-------------------|--------|
| NVIDIA DGX Spark (GB10) | SM121 | Verified |
| NVIDIA RTX 5090 | SM120 | Should work (same ISA) |
| NVIDIA RTX 5080 | SM120 | Should work |

## File Structure

```
Consumer-DeepGEMM/
│
│  ══════════════════════════════════════════════════════════════
│  ★ ACTUALLY DOES THE WORK (replaces Marlin FP4 MoE kernel)
│  ══════════════════════════════════════════════════════════════
│
├── csrc/
│   ├── cutlass_mxfp8_mxfp4_probe.cu  # ★★★ THE kernel:
│   │                            #   FP8×FP4 grouped GEMM via CUTLASS SM120 templates
│   │                            #   - Constructs pointer arrays for 256 MoE experts
│   │                            #   - Per-group SfAtom tile-interleaved scale reorder
│   │                            #   - Real CUTLASS initialize() + run() launch
│   │                            #   This single file is what fixes the garbage output.
│   ├── cutlass_sm120_probe.cu   # SM120 hardware capability probe
│   └── bindings.cpp             # PyBind11 entry — exports to consumer_deep_gemm._C
│
├── consumer_deep_gemm/
│   ├── gemm.py                  # ★ Python entry point for the fix:
│   │                            #   m_grouped_fp8_fp4_gemm_nt_contiguous()
│   │                            #   Routes to native _C kernel, falls back to Python dequant
│   ├── native.py                # ★ CUDA extension loader (tries _C first)
│   └── layout.py                # ★ Scale factor SfAtom reorder logic
│
├── scripts/
│   ├── install_in_vllm_container.sh    # ★ One-shot installer for vLLM Docker
│   └── patch_vllm_sm120_deep_gemm.py  # ★ Patches vLLM to route MoE FP4 through us
│
│  ══════════════════════════════════════════════════════════════
│  STUBS — only exist so vLLM's `import deep_gemm` doesn't crash
│  These are NOT used at runtime for actual computation.
│  vLLM's own kernels handle these paths on SM120 just fine.
│  ══════════════════════════════════════════════════════════════
│
├── consumer_deep_gemm/
│   ├── __init__.py              # (stub) Public API exports for import compat
│   ├── attention.py             # (stub) MQA/MLA — vLLM uses its own attention kernel
│   ├── hyperconnection.py       # (stub) HC pre-norm — jasl's Triton handles this
│   ├── einsum.py                # (stub) FP8 einsum — vLLM uses its own path
│   ├── mega.py                  # (stub) MegaMoE weight transforms
│   └── utils.py                 # (stub) SM count / TC util controls
│
├── deep_gemm/                   # (shim) `import deep_gemm` → consumer_deep_gemm
│   ├── __init__.py
│   └── utils/math.py
│
│  ══════════════════════════════════════════════════════════════
│  BUILD & TEST
│  ══════════════════════════════════════════════════════════════
│
├── scripts/
│   ├── build_native_sm120.sh           # Build native extension standalone
│   └── install_vllm_third_party_shim.py  # Write vllm.third_party.deep_gemm shim
│
├── tests/
│   ├── test_fp4_fallback.py     # Python fallback correctness
│   ├── test_native_abi.py       # Native extension ABI + GPU launch
│   ├── test_scale_reorder.py    # SfAtom scale layout reorder
│   ├── test_reorder_compare.py  # CPU vs GPU reorder consistency
│   └── test_gpu_reorder_debug.py
│
├── setup.py
└── README.md
```

## How It Works

### The Problem

vLLM on SM120+ for DeepSeek V4 Flash (MXFP4 MoE):
1. CUTLASS MXFP4 kernel **rejects** SM120+ (`Required capability: 100`)
2. Falls back to Marlin FP4 kernel
3. Marlin **accepts** SM120+ but computes wrong results → garbage output

### The Fix

Consumer-DeepGEMM replaces the entire GEMM computation path:

1. **New instructions**: `mma.sync.aligned` + block-scaled MX (SM120 native) instead of `tcgen05` (SM100 only)
2. **New data movement**: Manual pointer arrays instead of TMA hardware
3. **Scale factor reorder**: vLLM passes row-major `[M, K/128]` scales; CUTLASS SM120 needs SfAtom tile-interleaved layout — we do per-group reorder at launch time
4. **Python fallback**: If native launch fails, falls back to correct (but slow) Python dequant + matmul

### Performance

| Method | Speed | Correctness |
|--------|-------|-------------|
| vLLM Marlin (SM120+) | ~13 tok/s | ❌ Silent errors |
| Consumer-DeepGEMM | ~0.8 tok/s | ✅ All languages |
| jasl/vllm Triton fork | ~14 tok/s | ✅ Chinese, ⚠️ English residual |

Consumer-DeepGEMM is slow (0.8 tok/s) because each token requires 120 sequential CUTLASS kernel launches (60 layers × 2 FC) with CPU↔GPU sync between each. The value is **proving the fix works**, not daily use. For daily use, see [jasl's Triton fork](https://github.com/jasl/vllm/tree/ds4-sm120).

## Installation

### Quick (Python fallback only)

```bash
pip install .
```

### With CUDA native extension (inside vLLM container)

```bash
cd /path/to/Consumer-DeepGEMM
./scripts/install_in_vllm_container.sh
```

This script:
1. Builds `consumer_deep_gemm._C` for SM121a (CUTLASS grouped FP8×FP4 kernel)
2. Installs the `deep_gemm` compatibility shim
3. Writes `vllm.third_party.deep_gemm` → `consumer_deep_gemm`
4. Patches vLLM's DeepGEMM support checks to accept SM120/SM121

### Manual build

```bash
CONSUMER_DEEP_GEMM_BUILD_CUDA=1 \
CONSUMER_DEEP_GEMM_CUDA_ARCH=121a \
CUTLASS_PATH=/path/to/cutlass \
pip install .
```

### Verify

```bash
python3 -c "import consumer_deep_gemm as dg; print(dg.native_build_info())"
# {'available': True, 'cutlass_sm120_probe': True, 'cutlass_mxfp8_mxfp4_probe': True, 'arch': 'sm_121a'}
```

## Integration with vLLM

### What gets patched

| vLLM File | Change |
|-----------|--------|
| `platforms/cuda.py` | Add SM120 family to DeepGEMM support check |
| `fused_moe/experts/deep_gemm_moe.py` | Allow SM120 for MoE FP4 path |
| `kernels/linear/scaled_mm/deep_gemm.py` | **Block** SM120 for ordinary FP8 linear (only MoE FP4 uses Consumer-DeepGEMM) |
| `warmup/deep_gemm_warmup.py` | Skip FP8 linear warmup on SM120 |

### Why block ordinary FP8 linear?

Consumer-DeepGEMM only implements the MoE FP4 grouped GEMM correctly. vLLM's ordinary FP8 linear already has working non-DeepGEMM kernels on SM120. If we let DeepGEMM claims cover FP8 linear too, vLLM would route those calls through our Python fallback with incompatible scale layouts → crash.

### Running with vLLM (dual DGX Spark)

```bash
# Build image with Consumer-DeepGEMM baked in
# (or use pre-built: docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest)

HF_HOME=/path/to/models \
./launch-cluster.sh -n <host_cx7_ip>,<slave_cx7_ip> -t vllm-node-sm121-cdg exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --max-model-len 1000000 \
  --enforce-eager
```

## Requirements

- CUDA 12.8+ (CUDA 13.0+ recommended for SM121)
- PyTorch 2.11+ with CUDA support
- CUTLASS 3.x (bundled via DeepGEMM submodule, or set `CUTLASS_PATH`)
- ARM64 (DGX Spark is Grace CPU) or x86_64

## Related Projects

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — Original, SM90/SM100 only
- [jasl/vllm](https://github.com/jasl/vllm/tree/ds4-sm120) — Triton rewrite, 14 tok/s, recommended for daily use
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Docker build framework for DGX Spark
- [Deployment guide](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark) — Full dual-Spark deployment walkthrough

## License

Apache 2.0

---

**[中文说明 / Chinese Documentation](README_CN.md)**
