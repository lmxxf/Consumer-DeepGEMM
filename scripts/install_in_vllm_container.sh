#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

if command -v git >/dev/null 2>&1; then
  git config --global --add safe.directory "$(pwd)" || true
fi

"${PYTHON_BIN}" -m pip install -e .
./scripts/build_native_sm120.sh
"${PYTHON_BIN}" scripts/install_vllm_third_party_shim.py
"${PYTHON_BIN}" scripts/patch_vllm_sm120_deep_gemm.py

"${PYTHON_BIN}" - <<'PY'
import deep_gemm
import vllm.third_party.deep_gemm as vllm_deep_gemm

print("deep_gemm:", deep_gemm.native_build_info())
print("vllm.third_party.deep_gemm:", vllm_deep_gemm.native_build_info())
PY
