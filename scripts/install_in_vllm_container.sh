#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

if command -v git >/dev/null 2>&1; then
  git config --global --add safe.directory "$(pwd)" || true
fi

export CONSUMER_DEEP_GEMM_BUILD_CUDA=1
export CONSUMER_DEEP_GEMM_CUDA_ARCH="${CONSUMER_DEEP_GEMM_CUDA_ARCH:-121a}"
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

"${PYTHON_BIN}" -m pip install .
"${PYTHON_BIN}" scripts/install_vllm_third_party_shim.py
"${PYTHON_BIN}" scripts/patch_vllm_sm120_deep_gemm.py

"${PYTHON_BIN}" - <<'PY'
import deep_gemm
import vllm.third_party.deep_gemm as vllm_deep_gemm

print("deep_gemm:", deep_gemm.native_build_info())
print("vllm.third_party.deep_gemm:", vllm_deep_gemm.native_build_info())
PY
