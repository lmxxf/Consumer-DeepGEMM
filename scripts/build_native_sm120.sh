#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CONSUMER_DEEP_GEMM_BUILD_CUDA=1
export CONSUMER_DEEP_GEMM_CUDA_ARCH="${CONSUMER_DEEP_GEMM_CUDA_ARCH:-120a}"
if [[ -z "${CUTLASS_PATH:-}" ]]; then
  if [[ -d ../DeepGEMM/third-party/cutlass ]]; then
    export CUTLASS_PATH="$(realpath ../DeepGEMM/third-party/cutlass)"
  else
    export CUTLASS_PATH=/home/lmxxf/work/deepseek-v4-flash-deployment/DeepGEMM/third-party/cutlass
  fi
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

"${PYTHON_BIN}" setup.py build_ext --inplace
"${PYTHON_BIN}" - <<'PY'
import consumer_deep_gemm as dg
import deep_gemm
print(dg.native_build_info())
print("deep_gemm shim:", deep_gemm.__name__)
PY
